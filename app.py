from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

import streamlit as st
from ABC import ABC_Algorithm


def main():
    st.title("Machine Learning Parameter Tuning with ABC Algorithm")
    st.write("""
    This app uses the Artificial Bee Colony (ABC) algorithm to optimize the parameters of a 
    Support Vector Machine (SVM) classifier. The ABC algorithm searches for the best values 
    of the regularization parameter (C) and the kernel coefficient (gamma) to maximize 
    the classification accuracy.
    """)

    # User input for algorithm parameters
    st.sidebar.header("Algorithm Parameters")
    dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))
    pop_count = st.sidebar.number_input("Population Count", min_value=1, value=10)
    max_iterations = st.sidebar.number_input("Max Iterations", min_value=1, value=50)
    limit = st.sidebar.number_input("Limit", value=5)

    # Load selected dataset
    if dataset_name == "Iris":
        data = load_iris()
    elif dataset_name == "Breast Cancer":
        data = load_breast_cancer()
    elif dataset_name == "Wine":
        data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)

    def objective_function(params, X_train, y_train):
        C, gamma = params
        model = SVC(C=C, gamma=gamma)
        scores = cross_val_score(model, X_train, y_train, cv=5)  # 5-fold cross-validation
        accuracy = scores.mean()
        return -accuracy  # We negate accuracy because ABC minimizes the function

    # Run the algorithm
    if st.button("Run ABC Algorithm"):
        abc = ABC_Algorithm(
            objective_function=lambda params: objective_function(params, X_train, y_train),
            pop_count=pop_count,
            solution_size=2,
            lower_bound=[0.1, 0.01],
            upper_bound=[10, 1],
            max_iterations=max_iterations,
            limit=limit
        )
        best_solution, best_fitness, progress = abc.run()

        st.write(f"Best SVM Parameters: C = {best_solution[0]:.2f}, Gamma = {best_solution[1]:.2f}")
        st.write(f"Best Accuracy: {(-best_fitness):.4f}")

        # Plotting the accuracies over iterations
        accuracies = [-fitness for _, fitness in progress]
        st.line_chart(accuracies)


if __name__ == "__main__":
    main()
