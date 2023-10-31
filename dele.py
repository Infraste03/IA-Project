def train_and_evaluate(model, x_train, y_train, x_val, y_val):
    # Fit the model to the training data
    model.fit(x_train, y_train)

    # Evaluate the model on the training data
    train_score = model.score(x_train, y_train)

    # Evaluate the model on the validation data
    val_score = model.score(x_val, y_val)

    return train_score, val_score

from sklearn.feature_selection import mutual_info_regression
import seaborn as sns

# Calculate the mutual information score for every feature
mutual_info = mutual_info_regression(x_train, y_train)

k = 7
top_k_mi_scores = sorted(mutual_info, reverse=True)[:k]
top_k_features = [i for i, score in enumerate(mutual_info) if score in top_k_mi_scores]

#top_k_features = [25, 29, 30, 38, 40, 41, 61]

top_k_features_tensor = tf.constant(top_k_features, dtype=tf.int32)
#x_train_selected = x_train[:, top_k_features]
x_train_selected = tf.gather(x_train, top_k_features_tensor, axis=1)
x_val_selected = tf.gather(x_val, top_k_features_tensor, axis=1)

selected_features_df = pd.DataFrame(x_train_selected, columns=[f"Feature {i}" for i in range(1, k+1)])
selected_features_df["Target"] = y_train
sns.pairplot(selected_features_df, diag_kind="hist")


# Train and evaluate models using selected features
rf_mean_score, rf_test_score = train_and_evaluate(rf_model, x_train_selected, y_train, x_val_selected, y_val)
xgb_mean_score, xgb_test_score = train_and_evaluate(xgb_model, x_train_selected, y_train, x_val_selected, y_val)
#nn_mean_score, nn_test_score = train_and_evaluate(nn_model, x_train_selected, y_train, x_val_selected, y_val)

# Print mean score and test score for each model
print("RAAAAAAAAAAAAAAAAAA= {:.2f}, Test Score = {:.2f}".format(rf_mean_score, rf_test_score))
print("BBBBBBBBBBBBBBBBB= {:.2f}, Test Score = {:.2f}".format(xgb_mean_score, xgb_test_score))
#print("Neural Network: Mean Score = {:.2f}, Test Score = {:.2f}".format(nn_mean_score, nn_test_score))