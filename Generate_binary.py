from svm import do_svm
from Random_Forest import do_random_forest, do_final_rf
from Bagging import do_Bagging, do_final_bagging
from Knn import do_Knn, do_final_Knn


# Generate binaries for testing
def generate_binaty_for_testing():
    print("Random_forest:")
    do_random_forest()

    print("Bagging:")
    do_Bagging()

    print("Knn")
    do_Knn()


# Generate final binaries for competition submission.
def generate_final_binaries():
    print("Random_forest:")
    do_final_rf()

    print("Bagging:")
    do_final_bagging()

    print("Knn")
    do_final_Knn()


def main():
    generate_final_binaries()
    # generate_binaty_for_testing()


if __name__ == '__main__':
    main()
