"""Part 2: the controlled operating layer built on Part 1's intelligence.

Part 1 decides what is worth selling. Part 2 turns an approved decision into
listings, inventory, orders, settlements and measured profit -- then feeds the
measurements back so the next decision rests on evidence rather than
assumption.

Everything that touches the outside world goes through an approval gate, an
idempotency key and an explicit failure classification. Nothing here can move
money or publish a listing on its own.
"""

PART2_VERSION = "part2/1.0.0"

__all__ = ["PART2_VERSION"]
