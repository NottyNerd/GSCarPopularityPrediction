import pandas as pd
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

CSV_COLUMN_NAMES = ['buying_price','maintainence_cost','number_of_doors','number_of_seats',
                    'luggage_boot_size','safety_rating','popularity']
TEST_CSV_COLUMN_NAMES = ['buying_price','maintainence_cost','number_of_doors','number_of_seats',
                    'luggage_boot_size','safety_rating']
train_path ='data/train.csv'
test_path ='data/test.csv'

def main():
    result = load_data()
    print(len(result[1]))
    test_value = load_test_data()
    print(len(test_value))

    scaler = StandardScaler()
    scaler.fit(result[0])
    X_train = scaler.transform(result[0])
    test_value = scaler.transform(test_value)

    classify = tree.DecisionTreeClassifier()
    classify = classify.fit(X_train, result[1])
    predict_dtc = classify.predict(test_value)


    print("Result from using Decision Tree Classifier:" + str(predict_dtc))
    prediction = pd.DataFrame({'popularity': predict_dtc})
    prediction.to_csv("data/predictiondtc.csv", index=False, header=False)



    classify = MLPClassifier(hidden_layer_sizes=(500,500,500,500,), max_iter=1000, alpha=0.0001,
                     solver='lbfgs', verbose=10,  random_state=120, tol=0.000000001, warm_start=True, learning_rate_init=0.05 )
    classify = classify.fit(X_train, result[1])
    predict_linear_mlp = classify.predict(test_value)

    print('Using MLPClassifier LBFGS SOLVER:' + str(predict_linear_mlp))
    prediction = pd.DataFrame({'popularity': predict_linear_mlp})
    prediction.to_csv("data/prediction.csv", index=False, header=False)



def load_data(y_name='popularity'):
    """Returns the dataset as (train_x, train_y)."""

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    return (train_x, train_y)

def load_test_data():
    """Returns the dataset as (test_x)."""
    test = pd.read_csv(test_path, names=TEST_CSV_COLUMN_NAMES)
    test_x = test

    return (test_x)



if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run(main)
    main()