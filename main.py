"""Entry point for the AI Helpdesk Ticket system.

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


class HelpdeskApplication:
    """Coordinates CLI argument parsing and launches the selected application mode."""

    def __init__(self, argv: list[str] | None = None) -> None:
        self.args = self._parse_args(argv)

    @staticmethod
    def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="AI Helpdesk Ticket system")
        parser.add_argument(
            "--test",
            action="store_true",
            help="run the tests instead of starting the web app",
        )
        parser.add_argument(
            "--live",
            action="store_true",
            help="with --test, also run the live Gemini check (uses real API quota)",
        )
        parser.add_argument(
            "--demo",
            action="store_true",
            help="print dummy output to check ai_manager works (fake Gemini, no API key needed)",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=5000,
            help="web server port (default 5000)",
        )
        parser.add_argument(
            "--host",
            default="127.0.0.1",
            help="address to listen on (use 0.0.0.0 in Docker)",
        )
        parser.add_argument(
            "--no-browser",
            action="store_true",
            help="do not open the browser automatically",
        )
        return parser.parse_args(argv)

    def run_tests(self) -> None:
        # test_ai is imported first: it supplies a dummy API key when none is set.
        import test_ai
        import test_data_manager
        import test_io_manager
        import test_logic_manager

        try:
            test_ai.run_offline_tests()
        except AssertionError as error:
            sys.exit(f"Offline test FAILED: {error!r}")

        failures = (
            test_logic_manager.run_all_tests()
            + test_data_manager.run_all_tests()
            + test_io_manager.run_all_tests()
        )
        if failures:
            sys.exit(f"{failures} test(s) FAILED")

        if self.args.live:
            test_ai.run_live_check()

    @staticmethod
    def run_demo() -> None:
        import dummy_ai

        if not dummy_ai.run_demo():
            sys.exit(1)

    def run_web(self) -> None:
        try:
            from io_manager import FlaskIOManager
        except RuntimeError as error:
            sys.exit(str(error))

        server = FlaskIOManager(
            host=self.args.host,
            port=self.args.port,
            auto_open=not self.args.no_browser,
        )
        server.start()

    def execute(self) -> None:
        if self.args.demo:
            self.run_demo()
        elif self.args.test:
            self.run_tests()
        else:
            self.run_web()


if __name__ == "__main__":
    HelpdeskApplication().execute()