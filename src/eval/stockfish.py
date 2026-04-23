import logging
import re
import subprocess

logger = logging.getLogger(__name__)


class StockfishClient:
    def __init__(self, depth: int = 15):
        self.depth = depth
        self._process: subprocess.Popen[str] | None = None

    def get_eval(self, fen: str) -> float | None:
        return self._evaluate_fen(fen)

    def _evaluate_fen(self, fen: str) -> float | None:
        try:
            self._ensure_process()
            assert self._process is not None

            self._send_command(f"position fen {fen}")
            self._send_command("eval")
            stdout = self._read_until("Final evaluation", _timeout=5)

            cp_score = self._parse_final_evaluation(stdout)
            if cp_score is not None:
                return cp_score

            self._send_command(f"position fen {fen}")
            self._send_command(f"go depth {self.depth}")
            stdout = self._read_until("bestmove", _timeout=10)

            cp_score = self._parse_go_output(stdout)
            if cp_score is not None:
                return cp_score

            logger.warning(f"No score found for {fen}, stdout:\n{stdout}")
            return None

        except subprocess.TimeoutExpired:
            self.close()
            logger.warning(f"Stockfish timed out for {fen}")
            return None
        except Exception as e:
            self.close()
            logger.warning(f"Stockfish eval failed for {fen}: {e}")
            return None

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        self._process = subprocess.Popen(
            ["stockfish"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self._send_command("uci")
        self._read_until("uciok", _timeout=5)
        self._send_command("isready")
        self._read_until("readyok", _timeout=5)

    def _send_command(self, command: str) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Stockfish process is not available")
        self._process.stdin.write(f"{command}\n")
        self._process.stdin.flush()

    def _read_until(self, marker: str, _timeout: float) -> str:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Stockfish process is not available")

        lines: list[str] = []
        while True:
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("Stockfish closed stdout unexpectedly")
            lines.append(line)
            if marker in line:
                return "".join(lines)

    def _parse_final_evaluation(self, stdout: str) -> float | None:
        for line in stdout.split("\n"):
            if "Final evaluation" in line:
                if "none" in line:
                    return None
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "evaluation":
                        try:
                            return float(parts[i + 1]) * 100
                        except (IndexError, ValueError):
                            pass
        return None

    def _parse_go_output(self, stdout: str) -> float | None:
        mate_score = None
        cp_score = None

        for line in stdout.split("\n"):
            if "score mate" in line:
                match = re.search(r"score mate (-?\d+)", line)
                if match:
                    mate_score = int(match.group(1))
            elif "score cp" in line:
                match = re.search(r"score cp (-?\d+)", line)
                if match:
                    cp_score = float(match.group(1))

        if mate_score is not None:
            return 1000.0 - mate_score * 10 if mate_score > 0 else -1000.0 - mate_score * 10

        if cp_score is not None:
            return cp_score

        return self._parse_final_evaluation(stdout)

    def close(self) -> None:
        if self._process is None:
            return

        try:
            if self._process.poll() is None:
                self._send_command("quit")
                self._process.wait(timeout=1)
        except Exception:
            self._process.kill()
        finally:
            self._process = None


def get_stockfish_client(depth: int = 15) -> StockfishClient:
    return StockfishClient(depth=depth)
