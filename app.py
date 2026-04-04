"""
CraftsCircle ML Recommendation API
====================================
Flask REST API that serves real-time recommendations.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import os
import json
import time

app = Flask(__name__)
CORS(app)

db = None

def init_firebase():
    global db
    try:
        service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        if service_account_json:
            service_account_dict = json.loads(service_account_json)
            cred = credentials.Certificate(service_account_dict)
        else:
            cred = credentials.Certificate("serviceAccount.json")
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase connected!")
    except Exception as e:
        print(f"Firebase init error: {e}")

init_firebase()

cache = {
    "products_df": None,
    "orders": [],
    "tfidf_matrix": None,
    "tfidf_ids": [],
    "vectorizer": None,
    "co_purchase": {},
    "category_map": {},
    "last_updated": 0
}

CACHE_TTL = 30 * 60

def refresh_cache():
    global cache
    try:
        print("Refreshing data cache...")

        # Fetch products
        docs = db.collection("products").get()
        products = []
        for doc in docs:
            data = doc.to_dict()
            data["productId"] = doc.id
            # Normalize product_name → name
            if "product_name" in data:
                data["name"] = data["product_name"]
            products.append(data)
        products_df = pd.DataFrame(products) if products else pd.DataFrame()

        # Fetch orders
        order_docs = db.collection("orders").get()
        orders = []
        for doc in order_docs:
            data = doc.to_dict()
            data["orderId"] = doc.id
            orders.append(data)

        if products_df.empty:
            print("No products found!")
            return

        # Build TF-IDF
        def build_text(row):
            parts = []
            for field in ["name", "category", "description", "sub_category"]:
                val = row.get(field, "")
                if isinstance(val, list):
                    val = " ".join(val)
                if val:
                    parts.append(str(val).lower())
            return " ".join(parts)

        products_df["text"] = products_df.apply(build_text, axis=1)
        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        tfidf_matrix = vectorizer.fit_transform(products_df["text"].fillna(""))
        tfidf_ids = products_df["productId"].tolist()

        # Build co-purchase map
        user_products = defaultdict(set)
        for order in orders:
            uid = order.get("userId", "")
            items = order.get("items", [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        pid = item.get("productId") or item.get("id")
                        if pid and uid:
                            user_products[uid].add(pid)

        co_purchase = defaultdict(Counter)
        for uid, bought in user_products.items():
            pl = list(bought)
            for i in range(len(pl)):
                for j in range(len(pl)):
                    if i != j:
                        co_purchase[pl[i]][pl[j]] += 1

        # Build category map
        category_map = {}
        if "category" in products_df.columns:
            for cat, group in products_df.groupby("category"):
                if pd.notna(cat) and cat:
                    category_map[str(cat).lower()] = group["productId"].tolist()

        cache.update({
            "products_df": products_df,
            "orders": orders,
            "tfidf_matrix": tfidf_matrix,
            "tfidf_ids": tfidf_ids,
            "vectorizer": vectorizer,
            "co_purchase": co_purchase,
            "category_map": category_map,
            "last_updated": time.time()
        })
        print(f"Cache refreshed: {len(products)} products, {len(orders)} orders")

    except Exception as e:
        print(f"Cache refresh error: {e}")


def get_cache():
    if time.time() - cache["last_updated"] > CACHE_TTL or cache["products_df"] is None:
        refresh_cache()
    return cache


def fetch_products_by_ids(product_ids, limit=10):
    results = []
    for pid in product_ids[:limit]:
        try:
            doc = db.collection("products").document(pid).get()
            if doc.exists:
                data = doc.to_dict()
                data["productId"] = doc.id
                # Normalize fields
                if "product_name" in data:
                    data["name"] = data["product_name"]
                if "price" in data:
                    try:
                        data["price"] = float(str(data["price"]).replace(",", ""))
                    except:
                        data["price"] = 0.0
                results.append(data)
        except Exception as e:
            print(f"Error fetching product {pid}: {e}")
    return results


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "CraftsCircle ML API is running",
        "cache_age_minutes": round((time.time() - cache["last_updated"]) / 60, 1)
    })


@app.route("/trending", methods=["GET"])
def trending():
    try:
        limit = int(request.args.get("limit", 10))
        c = get_cache()
        orders = c["orders"]
        products_df = c["products_df"]

        product_counts = Counter()
        for order in orders:
            items = order.get("items", [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        pid = item.get("productId") or item.get("id")
                        if pid:
                            product_counts[pid] += 1

        if product_counts:
            top_ids = [pid for pid, _ in product_counts.most_common(limit)]
        else:
            top_ids = products_df["productId"].head(limit).tolist() if not products_df.empty else []

        products = fetch_products_by_ids(top_ids, limit)
        return jsonify({"type": "trending", "count": len(products), "products": products})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/similar/<product_id>", methods=["GET"])
def similar(product_id):
    try:
        limit = int(request.args.get("limit", 6))
        c = get_cache()
        tfidf_matrix = c["tfidf_matrix"]
        tfidf_ids = c["tfidf_ids"]

        if tfidf_matrix is None or product_id not in tfidf_ids:
            return jsonify({"type": "similar", "count": 0, "products": []})

        idx = tfidf_ids.index(product_id)
        scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
        similar_indices = np.argsort(scores)[::-1]
        similar_ids = [tfidf_ids[i] for i in similar_indices
                       if tfidf_ids[i] != product_id and scores[i] > 0][:limit]

        products = fetch_products_by_ids(similar_ids, limit)
        return jsonify({"type": "similar", "productId": product_id,
                        "count": len(products), "products": products})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/alsobought/<product_id>", methods=["GET"])
def also_bought(product_id):
    try:
        limit = int(request.args.get("limit", 6))
        c = get_cache()
        co_purchase = c["co_purchase"]

        if product_id not in co_purchase:
            return similar(product_id)

        rec_ids = [p for p, _ in co_purchase[product_id].most_common(limit)]
        products = fetch_products_by_ids(rec_ids, limit)
        return jsonify({"type": "alsobought", "productId": product_id,
                        "count": len(products), "products": products})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/category/<category>", methods=["GET"])
def by_category(category):
    try:
        limit = int(request.args.get("limit", 10))
        c = get_cache()
        category_map = c["category_map"]

        cat_key = category.lower().replace(" ", "_").replace("-", "_")
        product_ids = category_map.get(cat_key, [])
        if not product_ids:
            for key, ids in category_map.items():
                if cat_key in key or key in cat_key:
                    product_ids = ids
                    break

        products = fetch_products_by_ids(product_ids, limit)
        return jsonify({"type": "category", "category": category,
                        "count": len(products), "products": products})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/personalized/<uid>", methods=["GET"])
def personalized(uid):
    try:
        limit = int(request.args.get("limit", 10))
        c = get_cache()
        orders = c["orders"]
        tfidf_matrix = c["tfidf_matrix"]
        tfidf_ids = c["tfidf_ids"]
        co_purchase = c["co_purchase"]
        products_df = c["products_df"]

        user_bought = set()
        for order in orders:
            if order.get("userId") == uid:
                items = order.get("items", [])
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            pid = item.get("productId") or item.get("id")
                            if pid:
                                user_bought.add(pid)

        if not user_bought:
            return trending()

        candidates = Counter()
        for pid in user_bought:
            if tfidf_matrix is not None and pid in tfidf_ids:
                idx = tfidf_ids.index(pid)
                scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
                for i, score in enumerate(scores):
                    cid = tfidf_ids[i]
                    if cid not in user_bought and score > 0:
                        candidates[cid] += score * 2
            if pid in co_purchase:
                for rec_pid, count in co_purchase[pid].items():
                    if rec_pid not in user_bought:
                        candidates[rec_pid] += count

        rec_ids = [pid for pid, _ in candidates.most_common(limit)]

        if len(rec_ids) < limit and not products_df.empty:
            for pid in products_df["productId"].tolist():
                if pid not in user_bought and pid not in rec_ids:
                    rec_ids.append(pid)
                if len(rec_ids) >= limit:
                    break

        products = fetch_products_by_ids(rec_ids, limit)
        return jsonify({"type": "personalized", "userId": uid,
                        "count": len(products), "products": products})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/refresh", methods=["POST"])
def refresh():
    try:
        refresh_cache()
        return jsonify({"status": "ok", "message": "Cache refreshed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    refresh_cache()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)