"""Entry point for the AI Helpdesk Ticket system.

main.py only wires the modules together; the work happens in them:
    io_manager    - web page + routes + console output (serves index.html);
                    sends each ticket through the three managers below
    ai_manager    - security checks, redaction, Gemini call, validation
    logic_manager - priority score, P1-P5, department routing, review flag
    data_manager  - saves and loads tickets.json, filter query
    test_ai, test_logic_manager, test_data_manager - offline unit tests
                    (test_ai also has an optional live Gemini check)
    dummy_ai      - demo run with a fake Gemini (no key needed)

Usage:
    python main.py                  start the web app and open the browser
    python main.py --no-browser     start the web app without opening a browser
    python main.py --port 8000      use a different port
    python main.py --test           run all the offline unit tests
    python main.py --test --live    offline tests, then the live Gemini check
    python main.py --demo           dummy output: real pipeline, fake Gemini
"""

import argparse
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="AI Helpdesk Ticket system")
    parser.add_argument("--test", action="store_true",
                        help="run the tests instead of starting the web app")
    parser.add_argument("--live", action="store_true",
                        help="with --test, also run the live Gemini check "
                             "(uses real API quota)")
    parser.add_argument("--demo", action="store_true",
                        help="print dummy output to check ai_manager works "
                             "(fake Gemini, no API key or quota needed)")
    parser.add_argument("--port", type=int, default=5000,
                        help="web server port (default 5000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open the browser automatically")
    return parser.parse_args(argv)


def run_tests(live=False):
    # test_ai is imported first: it supplies a dummy API key when none is set.
    import test_ai
    import test_data_manager
    import test_logic_manager

    try:
        test_ai.run_offline_tests()
    except AssertionError as error:
        sys.exit(f"Offline test FAILED: {error!r}")

    failures = test_logic_manager.run_all_tests() + test_data_manager.run_all_tests()
    if failures:
        sys.exit(f"{failures} test(s) FAILED")

    if live:
        test_ai.run_live_check()


def run_demo():
    import dummy_ai

    if not dummy_ai.run_demo():
        sys.exit(1)


def run_web(port, auto_open):
    # io_manager imports ai_manager, which refuses to load without an API key.
    try:
        import io_manager
    except RuntimeError as error:
        sys.exit(str(error))
    io_manager.run_server(port=port, auto_open=auto_open)


def main(argv=None):
    args = parse_args(argv)
    if args.demo:
        run_demo()
    elif args.test:
        run_tests(live=args.live)
    else:
        run_web(port=args.port, auto_open=not args.no_browser)


if __name__ == "__main__":
    main()
