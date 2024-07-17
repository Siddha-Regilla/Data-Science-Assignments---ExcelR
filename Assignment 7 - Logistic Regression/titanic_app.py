# Save the trained model to a file
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)
