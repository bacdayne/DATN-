BASE_SCHEMA = """
{
  "menu": [
    {
      "nm": "",
      "cnt": "",
      "unitprice": "",
      "price": ""
    }
  ],
  "sub_total": {
    "subtotal_price": "",
    "discount_price": "",
    "tax_price": "",
    "service_price": ""
  },
  "total": {
    "total_price": "",
    "cashprice": "",
    "changeprice": "",
    "creditcardprice": ""
  }
}
"""


def classify_template_agent(ocr_text):
    text = ocr_text.lower()

    if any(k in text for k in ["ticket", "cinema", "xxi", "cgv"]):
        return "ticket_agent"

    if any(k in text for k in ["card payment", "debit", "visa", "credit card", "cc.", "edc"]):
        return "card_payment_agent"

    if any(k in text for k in ["pb1", "pb-1", "tax", "vat", "service", "service chrg"]):
        return "tax_service_agent"

    if any(k in text for k in ["cash", "tunai"]) and any(k in text for k in ["change", "changed", "kembali", "kembalian"]):
        return "cash_change_agent"

    if any(k in text for k in ["subtotal", "sub total", "subttl", "sub ttl"]):
        return "subtotal_agent"

    return "general_agent"


def build_rules(agent_name):
    common = """
Return this JSON format only:
{
  "menu": [{"nm": "", "cnt": "", "unitprice": "", "price": ""}],
  "sub_total": {"subtotal_price": "", "discount_price": "", "tax_price": "", "service_price": ""},
  "total": {"total_price": "", "cashprice": "", "changeprice": "", "creditcardprice": ""}
}

Mapping:
menu name -> menu.nm
quantity -> menu.cnt
unit price -> menu.unitprice
item price -> menu.price
subtotal -> sub_total.subtotal_price
discount -> sub_total.discount_price
tax / vat / pb1 -> sub_total.tax_price
service -> sub_total.service_price
total / grand total / amount due -> total.total_price
cash / tunai -> total.cashprice
change / kembalian -> total.changeprice
card / debit / visa / edc -> total.creditcardprice
"""

    if agent_name == "cash_change_agent":
        return common + """
Focus:
cash value belongs to total.cashprice
change value belongs to total.changeprice
total value belongs to total.total_price
"""

    if agent_name == "card_payment_agent":
        return common + """
Focus:
card payment value belongs to total.creditcardprice
card receipt usually has empty cashprice
"""

    if agent_name == "tax_service_agent":
        return common + """
Focus:
pb1 / tax / vat value belongs to sub_total.tax_price
service charge value belongs to sub_total.service_price
subtotal value belongs to sub_total.subtotal_price
"""

    if agent_name == "ticket_agent":
        return common + """
Focus:
ticket can be a menu item
edc / card value belongs to total.creditcardprice
"""

    if agent_name == "subtotal_agent":
        return common + """
Focus:
subtotal value belongs to sub_total.subtotal_price
product lines are before subtotal or total
"""

    return common


def get_prompt_by_agent(agent_name, ocr_text):
    rules = build_rules(agent_name)

    return f"""
You extract receipt OCR text into JSON.

{rules}

OCR:
{ocr_text}

JSON:
"""