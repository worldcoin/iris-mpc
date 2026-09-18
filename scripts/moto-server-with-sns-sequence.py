#!/usr/bin/env python3
"""Run Moto with the FIFO SNS SequenceNumber field emitted by AWS.

Moto 5.1.22 omits ``SequenceNumber`` from the SNS notification envelope sent
to subscribed SQS queues. Production AWS includes it for FIFO topics and the
iris-mpc server deliberately requires it. Keep the compatibility adjustment
at the emulator boundary instead of weakening production message parsing.
"""

from itertools import count
from threading import Lock

from moto.server import main
from moto.sns.models import Subscription


_original_get_post_data = Subscription.get_post_data
_sequence_by_message_id: dict[str, str] = {}
_next_sequence = count(1)
_sequence_lock = Lock()


def _get_post_data_with_sequence(self, message, message_id, subject, message_attributes=None):
    post_data = _original_get_post_data(
        self, message, message_id, subject, message_attributes
    )
    with _sequence_lock:
        sequence_number = _sequence_by_message_id.get(message_id)
        if sequence_number is None:
            # AWS FIFO sequence numbers are numeric strings whose lexical order
            # agrees with publication order. Use a fixed width for that same
            # property and reuse the value across all topic subscriptions.
            sequence_number = f"{next(_next_sequence):038d}"
            _sequence_by_message_id[message_id] = sequence_number
    post_data["SequenceNumber"] = sequence_number
    return post_data


Subscription.get_post_data = _get_post_data_with_sequence


if __name__ == "__main__":
    main()
