import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Features
features = ["Pclass", "Sex", "SibSp", "Parch"]

# Convert categorical values
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

# Target
y = train["Survived"]

# Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=1
)

# Train
model.fit(X, y)

# Predict
predictions = model.predict(X_test)

# Submission
output = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})

output.to_csv("submissions/submission.csv", index=False)

print("Submission file created successfully!")