#!/usr/bin/env python3
"""
ITP GUI - Interactive Theorem Proving Graphical User Interface
A web-based interface for interacting with Lean 4 proofs using Lean4SyncExecutor
"""

import sys
import os

# Add the parent directory to the path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import traceback
from typing import Optional, Dict, Any, List
import json

from itp_interface.tools.simple_lean4_sync_executor import SimpleLean4SyncExecutor
from itp_interface.lean_server.lean_context import ProofContext

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state to store the executor
executor_state: Dict[str, Any] = {
    'executor': None,
    'context_manager': None,
    'history': [],  # List of tactics executed
    'project_root': None,
    'file_path': None,
    'lemma_name': None
}


def get_debug_info() -> Dict[str, Any]:
    """Extract debug information from the Lean4SyncExecutor private variables"""
    if not executor_state['executor']:
        return {}

    executor = executor_state['executor']
    assert isinstance(executor, SimpleLean4SyncExecutor)

    # Format proof_context
    proof_context_info = None
    if executor.proof_context and executor.proof_context != ProofContext.empty():
        proof_context_info = {
            'num_goals': len(executor.proof_context.all_goals),
            'goals': [
                {
                    'hypotheses': [str(hyp) for hyp in goal.hypotheses],
                    'goal': str(goal.goal)
                }
                for goal in executor.proof_context.all_goals
            ]
        }

    debug_info = {
        'line_num': executor.line_num,
        'current_stmt': executor.current_stmt,
        'execution_complete': executor.execution_complete,
        'curr_lemma_name': executor.curr_lemma_name,
        'curr_lemma': executor.curr_lemma,
        '_last_tactics': executor._last_tactics,
        "_nested_have_counts": executor._nested_have_counts,
        "_nested_calc_counts": executor._nested_calc_counts,
        '_last_tactic_was_modified': executor._last_tactic_was_modified,

        # Requested private variables
        'proof_context': proof_context_info,
        '_proof_running': executor._proof_running,
        'lean_error_messages': executor.lean_error_messages,
        '_error_messages_since_last_thm': dict(executor._error_messages_since_last_thm),
        '_error_messages_so_far': list(executor._error_messages_so_far),

        # Other useful info
        'proof_start_idx': executor._proof_start_idx,
        'import_end_idx': executor._import_end_idx,
        'theorem_started': executor._theorem_started,
        'enforce_qed': executor._enforce_qed,
        'anon_theorem_count': executor._anon_theorem_count,

        # Debug traces and proof tactics
        'debug_enabled': executor.debug_enabled,
        'debug_traces': list(executor._debug_traces),
        'possible_proof_tactics': executor.possible_proof_tactics
    }

    return debug_info


def get_proof_state() -> Dict[str, Any]:
    """Get current proof state information"""
    if not executor_state['executor']:
        return {
            'initialized': False,
            'error': 'Executor not initialized'
        }

    executor = executor_state['executor']
    proof_context = executor.proof_context

    result = {
        'initialized': True,
        'lemma_name': executor.curr_lemma_name,
        'lemma_stmt': executor.curr_lemma,
        'is_in_proof_mode': executor.is_in_proof_mode(),
        'execution_complete': executor.execution_complete,
        'error_messages': executor.lean_error_messages,
        'goals': [],
        'history': executor_state['history']
    }

    if proof_context and proof_context != ProofContext.empty():
        for goal in proof_context.all_goals:
            goal_dict = {
                'hypotheses': [str(hyp) for hyp in goal.hypotheses],
                'goal': str(goal.goal)
            }
            result['goals'].append(goal_dict)

    return result


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')


@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the Lean4SyncExecutor with given parameters"""
    try:
        data = request.json
        project_root = data.get('project_root')
        file_path = data.get('file_path')
        lemma_name = data.get('lemma_name')

        if not all([project_root, file_path, lemma_name]):
            return jsonify({
                'success': False,
                'error': 'Missing required parameters: project_root, file_path, lemma_name'
            }), 400

        # Validate paths
        if not os.path.exists(project_root):
            return jsonify({
                'success': False,
                'error': f'Project root does not exist: {project_root}'
            }), 400

        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': f'File path does not exist: {file_path}'
            }), 400

        # Close existing executor if any
        if executor_state['context_manager']:
            try:
                executor_state['context_manager'].__exit__(None, None, None)
            except:
                pass

        # Initialize new executor
        executor = SimpleLean4SyncExecutor(
            project_root=project_root,
            main_file=file_path,
            timeout_in_sec=60,
            use_human_readable_proof_context=True,
            suppress_error_log=False,
            logger=logger
        )

        # Enter context manager
        executor.__enter__()

        # Enable debug mode
        executor.debug_enabled = True

        # Store in global state
        executor_state['executor'] = executor
        executor_state['context_manager'] = executor
        executor_state['project_root'] = project_root
        executor_state['file_path'] = file_path
        executor_state['lemma_name'] = lemma_name
        executor_state['history'] = []

        # Skip to the specified theorem
        try:
            executor._skip_to_theorem(lemma_name)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to find lemma "{lemma_name}": {str(e)}'
            }), 400

        # Get initial state
        state = get_proof_state()
        debug = get_debug_info()

        return jsonify({
            'success': True,
            'message': f'Initialized with lemma: {lemma_name}',
            'state': state,
            'debug': debug
        })

    except Exception as e:
        logger.error(f"Error initializing: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/run_tactic', methods=['POST'])
def run_tactic():
    """Run a tactic on the current proof state"""
    try:
        if not executor_state['executor']:
            return jsonify({
                'success': False,
                'error': 'Executor not initialized. Please initialize first.'
            }), 400

        data = request.json
        tactic = data.get('tactic', '')

        if not tactic or not tactic.strip():
            return jsonify({
                'success': False,
                'error': 'Tactic cannot be empty'
            }), 400

        executor = executor_state['executor']

        # Store current state before running
        prev_line_num = executor.line_num

        # Manually create a proof step iterator with the tactic
        # We need to inject this tactic into the executor
        old_iter = executor.main_file_iter

        # Create a simple iterator that yields the tactic as-is (no splitting)
        def tactic_iter():
            yield tactic
            # Then continue with the rest of the file
            while True:
                try:
                    yield next(old_iter)
                except StopIteration:
                    break

        executor.main_file_iter = tactic_iter()

        # Run the tactic (single execution)
        success = executor.run_next()

        # Add to history
        executor_state['history'].append({
            'tactic': tactic,
            'line_num': prev_line_num,
            'success': success,
            'errors': list(executor.lean_error_messages) if executor.lean_error_messages else []
        })

        # Get updated state
        state = get_proof_state()
        debug = get_debug_info()

        return jsonify({
            'success': True,
            'tactic_executed': tactic,
            'state': state,
            'debug': debug
        })

    except Exception as e:
        logger.error(f"Error running tactic: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get the current proof state"""
    try:
        state = get_proof_state()
        debug = get_debug_info()

        return jsonify({
            'success': True,
            'state': state,
            'debug': debug
        })

    except Exception as e:
        logger.error(f"Error getting state: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/validate', methods=['POST'])
def validate_proof():
    """Validate the current proof using lake lean (independent of REPL)"""
    try:
        if not executor_state['executor']:
            return jsonify({
                'success': False,
                'error': 'Executor not initialized. Please initialize first.'
            }), 400

        executor = executor_state['executor']

        # Get optional parameters from request
        data = request.json or {}
        timeout_sec = data.get('timeout_sec', 30)
        keep_temp_file = data.get('keep_temp_file', True)  # Default to keeping the file

        # Run validation
        validation_result = executor.validate_proof(
            timeout_sec=timeout_sec,
            keep_temp_file=keep_temp_file
        )

        return jsonify({
            'success': True,
            'validation': validation_result
        })

    except Exception as e:
        logger.error(f"Error validating proof: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/kill', methods=['POST'])
def kill_executor():
    """Kill/exit the executor and clean up resources"""
    try:
        if executor_state['context_manager']:
            logger.info("Killing executor and cleaning up resources")
            executor_state['context_manager'].__exit__(None, None, None)
            executor_state['executor'] = None
            executor_state['context_manager'] = None
            executor_state['history'] = []
            executor_state['project_root'] = None
            executor_state['file_path'] = None
            executor_state['lemma_name'] = None

            return jsonify({
                'success': True,
                'message': 'Executor killed and resources cleaned up'
            })
        else:
            return jsonify({
                'success': True,
                'message': 'No active executor to kill'
            })

    except Exception as e:
        logger.error(f"Error killing executor: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the executor to initial state"""
    try:
        if executor_state['context_manager']:
            executor_state['context_manager'].__exit__(None, None, None)

        executor_state['executor'] = None
        executor_state['context_manager'] = None
        executor_state['history'] = []

        return jsonify({
            'success': True,
            'message': 'Executor reset successfully'
        })

    except Exception as e:
        logger.error(f"Error resetting: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'initialized': executor_state['executor'] is not None
    })


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ITP GUI Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    logger.info(f"Starting ITP GUI on {args.host}:{args.port}")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    finally:
        # Cleanup on exit
        if executor_state['context_manager']:
            try:
                executor_state['context_manager'].__exit__(None, None, None)
            except:
                pass
