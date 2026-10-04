from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

PASTEL_COLORS = ["#8dd3c7", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69"]


@dataclass
class SimulationResult:
    protocol: str
    packet_count: int
    window_size: int
    loss_probability: float
    transmissions: int
    elapsed_slots: int

    @property
    def retransmissions(self) -> int:
        return self.transmissions - self.packet_count

    @property
    def overhead_percent(self) -> float:
        return 100.0 * self.retransmissions / self.packet_count


class UnreliableDataChannel:
    """
    Ненадежный канал передачи данных.
    В данной модели предполагается, что подтверждения доставляются надежно. 
    Пакет с повреждениями обрабатывается так же, как и потерянный пакет: он не доходит до получателя.
    """

    def __init__(self, loss_probability: float, rng: random.Random):
        if not 0.0 <= loss_probability < 1.0:
            raise ValueError("loss_probability must be in [0, 1)")
        self.loss_probability = loss_probability
        self.rng = rng

    def delivered(self) -> bool:
        return self.rng.random() >= self.loss_probability


class GBNReceiver:

    def __init__(self):
        self.expected_seq = 0

    def receive(self, seq: int) -> int:
        if seq == self.expected_seq:
            self.expected_seq += 1
        return self.expected_seq - 1


class SRReceiver:

    def __init__(self):
        self.base = 0
        self.buffer = set()

    def receive(self, seq: int) -> int:
        self.buffer.add(seq)
        while self.base in self.buffer:
            self.buffer.remove(self.base)
            self.base += 1
        return seq


class GBNSender:

    def __init__(self, packet_count: int, window_size: int):
        self.n = packet_count
        self.w = window_size
        self.base = 0
        self.next_seq = 0
        self.timer_start: Optional[int] = None

    @property
    def complete(self) -> bool:
        return self.base >= self.n

    def on_ack(self, ack: int, time_slot: int) -> None:
        if ack >= self.base:
            self.base = ack + 1
            if self.base == self.next_seq:
                self.timer_start = None
            else:
                self.timer_start = time_slot

    def handle_timeout(self, time_slot: int, timeout: int) -> None:
        if self.timer_start is not None and time_slot - self.timer_start >= timeout:
            self.next_seq = self.base
            self.timer_start = None

    def packet_to_send(self, time_slot: int) -> Optional[int]:
        if self.next_seq < self.n and self.next_seq < self.base + self.w:
            seq = self.next_seq
            self.next_seq += 1
            if self.timer_start is None and seq == self.base:
                self.timer_start = time_slot
            return seq
        return None


class SRSender:

    def __init__(self, packet_count: int, window_size: int):
        self.n = packet_count
        self.w = window_size
        self.base = 0
        self.next_seq = 0
        self.acked = [False] * packet_count
        self.last_send: List[Optional[int]] = [None] * packet_count

    @property
    def complete(self) -> bool:
        return self.base >= self.n

    def on_ack(self, ack: int) -> None:
        if 0 <= ack < self.n:
            self.acked[ack] = True
        while self.base < self.n and self.acked[self.base]:
            self.base += 1
        if self.next_seq < self.base:
            self.next_seq = self.base

    def packet_to_send(self, time_slot: int, timeout: int) -> Optional[int]:
        window_end = min(self.base + self.w, self.n)

        for seq in range(self.base, window_end):
            if self.acked[seq]:
                continue
            if self.last_send[seq] is not None and time_slot - self.last_send[seq] >= timeout:
                self.last_send[seq] = time_slot
                return seq

        if self.next_seq < window_end:
            seq = self.next_seq
            self.next_seq += 1
            self.last_send[seq] = time_slot
            return seq

        return None


class GBNSimulator:

    def __init__(
        self,
        packet_count: int,
        window_size: int,
        loss_probability: float,
        rng: random.Random,
        propagation_delay: int = 20,
        timeout: int = 50,
    ):
        if packet_count <= 0:
            raise ValueError("packet_count must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if timeout <= 2 * propagation_delay:
            raise ValueError("timeout must be larger than the round-trip propagation delay")

        self.n = packet_count
        self.w = window_size
        self.channel = UnreliableDataChannel(loss_probability, rng)
        self.receiver = GBNReceiver()
        self.propagation_delay = propagation_delay
        self.timeout = timeout

    def run(self) -> SimulationResult:
        sender = GBNSender(self.n, self.w)
        transmissions = 0
        time_slot = 0

        data_arrivals: Dict[int, List[int]] = {}
        ack_arrivals: Dict[int, List[int]] = {}
        max_slots = 10_000_000

        while not sender.complete:
            if time_slot > max_slots:
                raise RuntimeError("simulation did not converge, reduce loss probability")

            for seq in data_arrivals.pop(time_slot, []):
                ack = self.receiver.receive(seq)
                ack_time = time_slot + self.propagation_delay
                ack_arrivals.setdefault(ack_time, []).append(ack)

            for ack in ack_arrivals.pop(time_slot, []):
                sender.on_ack(ack, time_slot)

            sender.handle_timeout(time_slot, self.timeout)
            seq = sender.packet_to_send(time_slot)

            if seq is not None:
                transmissions += 1
                if self.channel.delivered():
                    arrival_time = time_slot + self.propagation_delay
                    data_arrivals.setdefault(arrival_time, []).append(seq)

            time_slot += 1

        return SimulationResult(
            protocol="GBN",
            packet_count=self.n,
            window_size=self.w,
            loss_probability=self.channel.loss_probability,
            transmissions=transmissions,
            elapsed_slots=time_slot,
        )


class SRSimulator:

    def __init__(
        self,
        packet_count: int,
        window_size: int,
        loss_probability: float,
        rng: random.Random,
        propagation_delay: int = 20,
        timeout: int = 50,
    ):
        if packet_count <= 0:
            raise ValueError("packet_count must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if timeout <= 2 * propagation_delay:
            raise ValueError("timeout must be larger than the round-trip propagation delay")

        self.n = packet_count
        self.w = window_size
        self.channel = UnreliableDataChannel(loss_probability, rng)
        self.receiver = SRReceiver()
        self.propagation_delay = propagation_delay
        self.timeout = timeout

    def run(self) -> SimulationResult:
        sender = SRSender(self.n, self.w)
        transmissions = 0
        time_slot = 0

        data_arrivals: Dict[int, List[int]] = {}
        ack_arrivals: Dict[int, List[int]] = {}
        max_slots = 10_000_000

        while not sender.complete:
            if time_slot > max_slots:
                raise RuntimeError("simulation did not converge, reduce loss probability")

            for seq in data_arrivals.pop(time_slot, []):
                ack = self.receiver.receive(seq)
                ack_time = time_slot + self.propagation_delay
                ack_arrivals.setdefault(ack_time, []).append(ack)

            for ack in ack_arrivals.pop(time_slot, []):
                sender.on_ack(ack)

            seq = sender.packet_to_send(time_slot, self.timeout)
            if seq is not None:
                transmissions += 1
                if self.channel.delivered():
                    arrival_time = time_slot + self.propagation_delay
                    data_arrivals.setdefault(arrival_time, []).append(seq)

            time_slot += 1

        return SimulationResult(
            protocol="SR",
            packet_count=self.n,
            window_size=self.w,
            loss_probability=self.channel.loss_probability,
            transmissions=transmissions,
            elapsed_slots=time_slot,
        )


def run_one(
    protocol: str,
    packet_count: int,
    window_size: int,
    loss_probability: float,
    seed: int,
    propagation_delay: int,
    timeout: int,
) -> SimulationResult:
    rng = random.Random(seed)
    if protocol == "GBN":
        sim = GBNSimulator(
            packet_count,
            window_size,
            loss_probability,
            rng,
            propagation_delay,
            timeout,
        )
    elif protocol == "SR":
        sim = SRSimulator(
            packet_count,
            window_size,
            loss_probability,
            rng,
            propagation_delay,
            timeout,
        )
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    return sim.run()


def experiment(
    packet_count: int,
    windows: List[int],
    loss_probabilities: List[float],
    repeats: int,
    base_seed: int,
    propagation_delay: int,
    timeout: int,
) -> List[dict]:
    rows: List[dict] = []

    for protocol_index, protocol in enumerate(("GBN", "SR")):
        for window_index, window in enumerate(windows):
            for loss_index, p in enumerate(loss_probabilities):
                overhead_values: List[float] = []
                transmission_values: List[int] = []
                time_values: List[int] = []

                for repeat in range(repeats):
                    seed = (
                        base_seed
                        + protocol_index * 10_000_000
                        + window_index * 1_000_000
                        + loss_index * 10_000
                        + repeat
                    )
                    result = run_one(
                        protocol,
                        packet_count,
                        window,
                        p,
                        seed,
                        propagation_delay,
                        timeout,
                    )
                    overhead_values.append(result.overhead_percent)
                    transmission_values.append(result.transmissions)
                    time_values.append(result.elapsed_slots)

                rows.append(
                    {
                        "protocol": protocol,
                        "window": window,
                        "loss_probability": p,
                        "loss_percent": 100.0 * p,
                        "mean_overhead_percent": mean(overhead_values),
                        "std_overhead_percent": stdev(overhead_values) if repeats > 1 else 0.0,
                        "mean_transmissions": mean(transmission_values),
                        "mean_elapsed_slots": mean(time_values),
                        "repeats": repeats,
                    }
                )
    return rows


def save_csv(rows: List[dict], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: List[dict], protocol: str, windows: List[int], output: Path) -> None:
    plt.figure(figsize=(8, 5))
    for index, window in enumerate(windows):
        selected = [r for r in rows if r["protocol"] == protocol and r["window"] == window]
        selected.sort(key=lambda r: r["loss_percent"])
        x = [r["loss_percent"] for r in selected]
        y = [r["mean_overhead_percent"] for r in selected]
        plt.plot(
            x,
            y,
            marker="o",
            color=PASTEL_COLORS[index % len(PASTEL_COLORS)],
            label=f"WIN = {window}",
        )

    plt.xlabel("Вероятность потери пакета, %")
    plt.ylabel("Накладные расходы, %")
    plt.title(f"{protocol}: зависимость накладных расходов от вероятности потери")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> None:
    packet_count = 200
    repeats = 50
    base_seed = 20260914
    propagation_delay = 20
    timeout = 50
    windows = [1, 4, 8, 16]
    loss_percent = list(range(0, 51, 5))
    output = Path("results")

    output.mkdir(parents=True, exist_ok=True)

    loss_probabilities = [x / 100.0 for x in loss_percent]
    rows = experiment(
        packet_count=packet_count,
        windows=windows,
        loss_probabilities=loss_probabilities,
        repeats=repeats,
        base_seed=base_seed,
        propagation_delay=propagation_delay,
        timeout=timeout,
    )

    save_csv(rows, output / "results.csv")
    plot(rows, "GBN", windows, output / "gbn_overhead.png")
    plot(rows, "SR", windows, output / "sr_overhead.png")

    print(f"Saved: {output / 'results.csv'}")
    print(f"Saved: {output / 'gbn_overhead.png'}")
    print(f"Saved: {output / 'sr_overhead.png'}")


if __name__ == "__main__":
    main()
