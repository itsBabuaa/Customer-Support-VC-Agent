"""Dummy API layer — laptop order database + product catalog."""

PRODUCTS = {
    "PRO-16": {
        "name": "NovaPro 16",
        "category": "Professional",
        "cpu": "Intel Core i9-14900HX",
        "ram": "32GB DDR5",
        "storage": "1TB NVMe SSD",
        "gpu": "NVIDIA RTX 4070",
        "display": '16" 2560x1600 165Hz IPS',
        "battery": "90Wh, ~8 hours",
        "weight": "2.1 kg",
        "price": 1899,
        "warranty": "2 years",
    },
    "AIR-14": {
        "name": "NovaAir 14",
        "category": "Ultrabook",
        "cpu": "Intel Core Ultra 7 155H",
        "ram": "16GB LPDDR5x",
        "storage": "512GB NVMe SSD",
        "gpu": "Intel Arc integrated",
        "display": '14" 2880x1800 120Hz OLED',
        "battery": "72Wh, ~12 hours",
        "weight": "1.2 kg",
        "price": 1299,
        "warranty": "2 years",
    },
    "STU-15": {
        "name": "NovaBook 15",
        "category": "Student / Budget",
        "cpu": "AMD Ryzen 5 7535HS",
        "ram": "16GB DDR5",
        "storage": "512GB NVMe SSD",
        "gpu": "AMD Radeon integrated",
        "display": '15.6" 1920x1080 60Hz IPS',
        "battery": "56Wh, ~9 hours",
        "weight": "1.7 kg",
        "price": 699,
        "warranty": "1 year",
    },
    "GAM-17": {
        "name": "NovaForce 17",
        "category": "Gaming",
        "cpu": "AMD Ryzen 9 7945HX",
        "ram": "32GB DDR5",
        "storage": "2TB NVMe SSD",
        "gpu": "NVIDIA RTX 4080",
        "display": '17.3" 2560x1440 240Hz IPS',
        "battery": "99Wh, ~5 hours",
        "weight": "2.8 kg",
        "price": 2499,
        "warranty": "2 years",
    },
}

ORDERS = {
    "NLT-10001": {
        "customer": "[name]", "email": "[email]", "phone": "[phone_number]",
        "items": [{"product": "NovaPro 16", "sku": "PRO-16", "qty": 1, "price": 1899}],
        "total": 1899, "payment": "Credit Card",
        "status": "Delivered", "order_date": "2026-02-10",
        "ship_date": "2026-02-12", "delivery_date": "2026-02-15",
        "tracking": "1Z999AA10123456784",
    },
    "NLT-10002": {
        "customer": "[name]", "email": "[email]", "phone": "[phone_number]",
        "items": [
            {"product": "NovaAir 14", "sku": "AIR-14", "qty": 1, "price": 1299},
            {"product": "USB-C Hub", "sku": "ACC-HUB", "qty": 1, "price": 49},
        ],
        "total": 1348, "payment": "PayPal",
        "status": "Shipped", "order_date": "2026-02-20",
        "ship_date": "2026-02-22", "delivery_date": None,
        "tracking": "9400111899223100012",
    },
    "NLT-10003": {
        "customer": "[name]", "email": "[email]", "phone": "[phone_number]",
        "items": [{"product": "NovaForce 17", "sku": "GAM-17", "qty": 1, "price": 2499}],
        "total": 2499, "payment": "Credit Card",
        "status": "Processing", "order_date": "2026-02-28",
        "ship_date": None, "delivery_date": None, "tracking": None,
    },
    "NLT-10004": {
        "customer": "[name]", "email": "[email]", "phone": "[phone_number]",
        "items": [{"product": "NovaBook 15", "sku": "STU-15", "qty": 2, "price": 699}],
        "total": 1398, "payment": "Debit Card",
        "status": "Delivered", "order_date": "2026-02-05",
        "ship_date": "2026-02-07", "delivery_date": "2026-02-10",
        "tracking": "7489201472390184",
    },
    "NLT-10005": {
        "customer": "[name]", "email": "[email]", "phone": "[phone_number]",
        "items": [{"product": "NovaPro 16", "sku": "PRO-16", "qty": 1, "price": 1899}],
        "total": 1899, "payment": "Credit Card",
        "status": "Return Requested", "order_date": "2026-02-15",
        "ship_date": "2026-02-17", "delivery_date": "2026-02-20",
        "tracking": "1Z999BB20987654321",
    },
}


def lookup_order(order_id: str) -> str:
    order_id = order_id.strip().upper()
    order = ORDERS.get(order_id)
    if not order:
        return f"Order {order_id} not found. Please check the order number."
    items_str = ", ".join(f"{i['product']} x{i['qty']}" for i in order["items"])
    lines = [
        f"Order: {order_id}",
        f"Items: {items_str}",
        f"Total: ${order['total']:,}",
        f"Payment: {order['payment']}",
        f"Status: {order['status']}",
    ]
    if order["tracking"]:
        lines.append(f"Tracking: {order['tracking']}")
    if order["delivery_date"]:
        lines.append(f"Delivered: {order['delivery_date']}")
    elif order["ship_date"]:
        lines.append(f"Shipped: {order['ship_date']}")
    else:
        lines.append(f"Ordered: {order['order_date']}")
    return "\n".join(lines)


def lookup_by_phone(phone: str) -> str:
    phone = phone.strip().replace(" ", "")
    matches = [(oid, o) for oid, o in ORDERS.items() if o["phone"] == phone]
    if not matches:
        return f"No orders found for phone number {phone}."
    return "\n".join(f"{oid}: {o['status']}" for oid, o in matches)


def lookup_by_email(email: str) -> str:
    email = email.strip().lower()
    matches = [(oid, o) for oid, o in ORDERS.items() if o["email"].lower() == email]
    if not matches:
        return f"No orders found for email {email}."
    return "\n".join(f"{oid}: {o['status']}" for oid, o in matches)


def get_product_info(sku: str) -> str:
    sku = sku.strip().upper()
    product = PRODUCTS.get(sku)
    if not product:
        return f"Product {sku} not found."
    lines = [
        f"{product['name']} ({product['category']})",
        f"CPU: {product['cpu']}",
        f"RAM: {product['ram']} | Storage: {product['storage']}",
        f"GPU: {product['gpu']}",
        f"Display: {product['display']}",
        f"Battery: {product['battery']} | Weight: {product['weight']}",
        f"Price: ${product['price']} | Warranty: {product['warranty']}",
    ]
    return "\n".join(lines)


def list_products() -> str:
    lines = []
    for sku, p in PRODUCTS.items():
        lines.append(f"{p['name']} (${p['price']}) - {p['category']}")
    return "\n".join(lines)
