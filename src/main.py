from data_exploration import explore_data
from preprocessing import preprocess_data, split_features_target
from visualization import create_plots
from model import train_model
from evaluation import evaluate_model


def main():
    df = explore_data("data/spotify.csv")
    df = preprocess_data(df, add_target=True)
    create_plots(df)
    X_train, X_test, y_train, y_test = split_features_target(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()