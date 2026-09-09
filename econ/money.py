"""Money primitives and the cost components economics is assembled from.

Every amount is ``Decimal``. Floats are refused at the database boundary and
must not appear here either: a rounding error in money arithmetic surfaces
only once the numbers are already wrong, which is the most expensive time to
find it.

Rounding policy: intermediate arithmetic keeps four decimal places, and only
values presented or persisted as currency are quantised to two. Rounding at
every step compounds; rounding once at the edge does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

ZERO = Decimal("0")
HUNDRED = Decimal("100")
CURRENCY_PLACES = Decimal("0.01")
WORKING_PLACES = Decimal("0.0001")


def money(value: Any) -> Decimal:
    """Coerce to Decimal at working precision, refusing floats.

    A float here would silently reintroduce binary rounding into a system that
    is careful about it everywhere else, so it is rejected rather than
    converted.
    """
    if isinstance(value, float):
        raise TypeError(
            "refusing to build a monetary value from a float; pass Decimal or str"
        )
    return Decimal(str(value)).quantize(WORKING_PLACES, rounding=ROUND_HALF_UP)


def rupees(value: Any) -> Decimal:
    """Quantise to two places for presentation or persistence as currency."""
    return money(value).quantize(CURRENCY_PLACES, rounding=ROUND_HALF_UP)


RUPEE_SIGN = "\u20b9"


def format_inr(value: Any) -> str:
    """Render an amount as currency, with the rupee sign and separators.

    Reports quote money, and money in this domain is rupees. Rendering bare
    decimals leaves the reader to guess the unit, so every amount that reaches
    a human goes through here.

    The rupee sign has no cp1252 code point, so anything printing this must
    have a UTF-8 output path -- ``main.py`` reconfigures stdio for exactly
    this reason. The currency is domain, not decoration; the encoding is what
    adapts.
    """
    amount = rupees(value)
    sign = "-" if amount < 0 else ""
    whole, _, fraction = f"{abs(amount):.2f}".partition(".")
    # Indian digit grouping: last three, then pairs (12,34,567.89).
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join([*groups, tail])
    return f"{sign}{RUPEE_SIGN}{whole}.{fraction}"


def pct(value: Any) -> Decimal:
    """A percentage as a Decimal, e.g. ``18`` for 18%."""
    if isinstance(value, float):
        raise TypeError("refusing to build a percentage from a float")
    return Decimal(str(value))


def apply_pct(base: Decimal, percentage: Decimal) -> Decimal:
    """``base`` multiplied by ``percentage``%."""
    return money(base * pct(percentage) / HUNDRED)


def safe_div(numerator: Decimal, denominator: Decimal) -> Decimal | None:
    """Divide, returning None rather than raising on a zero denominator.

    A ratio against zero is undefined, not zero. Returning None forces the
    caller to decide what an undefined margin means instead of silently
    reporting 0%.
    """
    if denominator == ZERO:
        return None
    return money(numerator / denominator)


# --------------------------------------------------------------------------
# GST and input tax credit
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GstSplit:
    """A GST-inclusive amount decomposed into taxable value and tax."""

    gross_inr: Decimal
    taxable_inr: Decimal
    tax_inr: Decimal
    rate_pct: Decimal


def split_inclusive(gross: Decimal, rate_pct: Decimal) -> GstSplit:
    """Decompose a GST-inclusive amount."""
    gross = money(gross)
    rate = pct(rate_pct)
    taxable = money(gross * HUNDRED / (HUNDRED + rate)) if rate != ZERO else gross
    return GstSplit(gross, taxable, money(gross - taxable), rate)


def add_exclusive(net: Decimal, rate_pct: Decimal) -> GstSplit:
    """Add GST to a tax-exclusive amount."""
    net = money(net)
    tax = apply_pct(net, rate_pct)
    return GstSplit(money(net + tax), net, tax, pct(rate_pct))


def input_tax_credit(
    supplier_cost: Decimal,
    *,
    gst_rate_pct: Decimal,
    cost_is_gst_inclusive: bool | None,
    tax_invoice_available: bool,
) -> Decimal:
    """Recoverable input tax credit on a purchase.

    Zero without a tax invoice. A retail receipt carries no GSTIN, so the tax
    paid on that purchase is unrecoverable and the full amount is a cost --
    frequently the difference between a viable SKU and a dead one, which is
    why this is computed and reported as its own line rather than folded into
    the cost.
    """
    if not tax_invoice_available:
        # No invoice, no credit -- and inclusivity is moot, so an unknown
        # inclusivity flag is harmless here rather than blocking.
        return ZERO
    if cost_is_gst_inclusive is None:
        raise ValueError(
            "cannot compute input tax credit: the catalogue did not state "
            "whether the quoted cost is GST-inclusive, and assuming either way "
            "misstates the recoverable tax. Record it on the supplier offer."
        )
    if cost_is_gst_inclusive:
        return split_inclusive(supplier_cost, gst_rate_pct).tax_inr
    return apply_pct(supplier_cost, gst_rate_pct)


# --------------------------------------------------------------------------
# Marketplace fees
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeeBreakdown:
    """Marketplace fees on one order, itemised."""

    commission_inr: Decimal = ZERO
    fixed_fee_inr: Decimal = ZERO
    collection_fee_inr: Decimal = ZERO
    shipping_inr: Decimal = ZERO
    gst_on_fees_inr: Decimal = ZERO
    gst_on_fees_recoverable_inr: Decimal = ZERO
    notes: list[str] = field(default_factory=list)

    @property
    def total_before_gst_inr(self) -> Decimal:
        return money(
            self.commission_inr
            + self.fixed_fee_inr
            + self.collection_fee_inr
            + self.shipping_inr
        )

    @property
    def net_cost_inr(self) -> Decimal:
        """Fees actually borne, after recovering GST charged on them."""
        return money(
            self.total_before_gst_inr
            + self.gst_on_fees_inr
            - self.gst_on_fees_recoverable_inr
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "commission_inr": str(rupees(self.commission_inr)),
            "fixed_fee_inr": str(rupees(self.fixed_fee_inr)),
            "collection_fee_inr": str(rupees(self.collection_fee_inr)),
            "shipping_inr": str(rupees(self.shipping_inr)),
            "gst_on_fees_inr": str(rupees(self.gst_on_fees_inr)),
            "net_cost_inr": str(rupees(self.net_cost_inr)),
        }


class EconomicsDataUnavailable(ValueError):
    """A required economics input is absent from the reference data.

    Raised instead of substituting zero. Every one of these inputs is a cost:
    a missing commission, collection fee, GST-on-fees or return rate that
    silently became 0% understated cost and therefore overstated profit, which
    is the one direction of error that turns a loss-making SKU into an
    apparent opportunity. An explicitly configured zero is a different thing
    and is honoured.
    """


def _require(payload: dict[str, Any], key: str, document: str) -> Any:
    """Read a required reference value, refusing to invent one.

    ``None`` and absence are both refused; an explicit ``0`` passes through,
    because a rate card stating a zero fee is making a real claim.
    """
    if key not in payload or payload[key] is None:
        raise EconomicsDataUnavailable(
            f"{document} does not define {key!r}. It is not defaulted to zero: "
            "an unstated cost understates landed economics and overstates "
            "profit. State it explicitly, including zero if that is correct."
        )
    return payload[key]


def _band_lookup(
    bands: list[dict[str, Any]], value: Decimal, key: str, *, document: str = "rate card"
) -> Decimal:
    """Smallest band whose ceiling covers ``value``; the last band otherwise."""
    if not bands:
        raise EconomicsDataUnavailable(
            f"{document} defines no {key} bands. An empty band table is not a "
            "zero fee; state a band with fee 0 if the fee is genuinely nil."
        )
    ordered = sorted(bands, key=lambda b: Decimal(str(b[key])))
    for band in ordered:
        if value <= Decimal(str(band[key])):
            return money(band["fee"])
    return money(ordered[-1]["fee"])


def compute_fees(
    *,
    selling_price_inr: Decimal,
    category: str,
    weight_g: int,
    rate_card: dict[str, Any],
    registered_for_gst: bool = True,
) -> FeeBreakdown:
    """Marketplace fees for one order, from a pinned rate card.

    Reads only from the supplied rate card payload; no figure is hardcoded
    here, so changing a fee schedule is a data change with a version, not a
    code change.
    """
    price = money(selling_price_inr)
    notes: list[str] = []

    commission_table = _require(rate_card, "commission_pct_by_category", "rate card")
    if category in commission_table:
        commission_pct = commission_table[category]
    elif "default" in commission_table:
        commission_pct = commission_table["default"]
    else:
        raise EconomicsDataUnavailable(
            f"rate card has no commission slab for {category!r} and no default "
            "slab. A missing commission silently became 0%, which overstates "
            "margin on every order."
        )
    if category not in commission_table:
        notes.append(f"no commission slab for category {category!r}; used default")
    commission = apply_pct(price, commission_pct)

    fixed = _band_lookup(
        _require(rate_card, "fixed_fee_bands", "rate card"), price, "max_price"
    )
    collection = apply_pct(price, _require(rate_card, "collection_fee_pct", "rate card"))
    shipping = _band_lookup(
        _require(rate_card, "shipping_by_weight_g", "rate card"),
        Decimal(weight_g),
        "max_weight_g",
    )

    subtotal = money(commission + fixed + collection + shipping)
    gst_on_fees = apply_pct(subtotal, _require(rate_card, "gst_on_fees_pct", "rate card"))
    # GST charged on marketplace fees is recoverable only by a registered
    # seller; for anyone else it is a real cost.
    recoverable = gst_on_fees if registered_for_gst else ZERO
    if not registered_for_gst:
        notes.append("not GST-registered: GST on marketplace fees is unrecoverable")

    return FeeBreakdown(
        commission_inr=commission,
        fixed_fee_inr=fixed,
        collection_fee_inr=collection,
        shipping_inr=shipping,
        gst_on_fees_inr=gst_on_fees,
        gst_on_fees_recoverable_inr=recoverable,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Returns and RTO
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReturnCost:
    """Expected per-shipped-order cost of returns and RTO."""

    return_rate_pct: Decimal
    rto_rate_pct: Decimal
    cost_per_return_inr: Decimal
    unsaleable_write_down_inr: Decimal
    expected_cost_per_order_inr: Decimal
    delivered_fraction: Decimal

    def as_dict(self) -> dict[str, str]:
        return {
            "return_rate_pct": str(self.return_rate_pct),
            "rto_rate_pct": str(self.rto_rate_pct),
            "expected_cost_per_order_inr": str(rupees(self.expected_cost_per_order_inr)),
            "delivered_fraction": str(self.delivered_fraction),
        }


def compute_return_cost(
    *,
    landed_cost_inr: Decimal,
    benchmark: dict[str, Any],
    return_rate_multiplier: Decimal = Decimal("1"),
) -> ReturnCost:
    """Expected return cost amortised across every shipped order.

    Returns are charged per *shipped* order rather than per delivered one,
    because the logistics cost and the write-down are incurred whether or not
    the sale sticks. Reporting margin per delivered order would flatter every
    SKU by the return rate.
    """
    landed = money(landed_cost_inr)
    return_rate = money(
        pct(_require(benchmark, "return_rate_pct", "return benchmark"))
        * return_rate_multiplier
    )
    rto_rate = money(
        pct(_require(benchmark, "rto_rate_pct", "return benchmark"))
        * return_rate_multiplier
    )
    per_return = money(_require(benchmark, "cost_per_return_inr", "return benchmark"))
    unsaleable_pct = pct(_require(benchmark, "unsaleable_share_pct", "return benchmark"))

    combined_rate = min(money(return_rate + rto_rate), HUNDRED)
    write_down = apply_pct(landed, unsaleable_pct)
    expected = apply_pct(money(per_return + write_down), combined_rate)

    return ReturnCost(
        return_rate_pct=return_rate,
        rto_rate_pct=rto_rate,
        cost_per_return_inr=per_return,
        unsaleable_write_down_inr=write_down,
        expected_cost_per_order_inr=expected,
        delivered_fraction=money((HUNDRED - combined_rate) / HUNDRED),
    )


__all__ = [
    "CURRENCY_PLACES",
    "HUNDRED",
    "RUPEE_SIGN",
    "ZERO",
    "EconomicsDataUnavailable",
    "FeeBreakdown",
    "GstSplit",
    "ReturnCost",
    "add_exclusive",
    "apply_pct",
    "compute_fees",
    "compute_return_cost",
    "format_inr",
    "input_tax_credit",
    "money",
    "pct",
    "rupees",
    "safe_div",
    "split_inclusive",
]
