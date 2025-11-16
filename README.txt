NOTE TO THE READER:
    This is a write-up document detailing the various functions required for the 'Project 3'
assigment under the 'Computer Science 422/622: Machine Learning' course.    
    
    Instructions and the 'helpers.py' file has been provided by Dr. Emily Hand at the
University of Nevada, Reno. All code within the 'binary_Classification.py' file, as well
as the 'multi-class_Classification.py' file, is that of my own.



Binary Classification Write-Up:
    1. [w, b, S] = svm_train_brute(training_data)
    2. dist = distance_point_to_hyperplane(pt, w, b)
    3. margin = compute_margin(data, w, b)
    4. y = svm_test_brute(w, b, x)



Multi-Class Classification Write-Up:
    1. [W, B] = svm_train_multiclass(training_data)
    2. y = svm_test_multiclass(W, B, x)
    3. plot_data_and_boundaries(data, W, B)