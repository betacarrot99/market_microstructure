def evaluate_individual_predictions(df, now, prediction_queue, signal_accuracy, threshold=0.000001):
    current_price = df["price"].iloc[-1]
    updated = []

    for pred in prediction_queue:
        if now >= pred["eval_time"]:
            threshold_val = pred["price"] * threshold
            price_diff = current_price - pred["price"]

            if price_diff > threshold_val:
                truth = 1
            elif price_diff < -threshold_val:
                truth = -1
            else:
                truth = 0

            for name, val in pred["signals"].items():
                signal_accuracy[name].append(val == truth)
        else:
            updated.append(pred)

    prediction_queue.clear()
    prediction_queue.extend(updated)
