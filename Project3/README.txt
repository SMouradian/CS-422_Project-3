NOTE TO THE READER:
        This is a write-up document detailing the various functions required for the 'Project 3'
    ssigment under the 'Computer Science 422/622: Machine Learning' course.    
    
        Instructions and the 'helpers.py' file has been provided by Dr. Emily Hand at the
    University of Nevada, Reno. All code within the 'binary_Classification.py' file, as well
    as the 'multi-class_Classification.py' file, is that of my own.



Binary Classification Write-Up:
    1. [w, b, S] = svm_train_brute(training_data)
            This function takes the specified training data that is initially set by the user
        within the 'generate_training_data_binary' function that is provided in the 'helpers.py'
        file. After the training data is plotted using the 'plot_training_data_and_binary'
        function (also from the 'helpers.py' file), this function creates a variable for our
        wight vector [w], our bias [b], and the array of support vectors [S]. These three variables
        are then used by the function to create a decision boundary for the training data, which
        utilizes a calculated margin that the function also creates. After all of this is created,
        the three variables: [w], [b], and [S] are returned. If the margin is not calculated properly,
        or if the training data is not being trained by the function, then the function stops.


    2. dist = distance_point_to_hyperplane(pt, w, b)

            This function takes the [w] and [b] values that were returned from the 'svm_train_brute'
        function, and then uses a data point [pt] specified by the user (I simply chose (1, 1) as my
        data point if you look at my test code at the very bottom of the 'svm.py' file). All it does
        is it calculates the distance between the user's specified data point and the hyperplane that
        that separates the data labels (positive and negative). It then returns this as a quotient.


    3. margin = compute_margin(data, w, b)

            This function computes the margin that is used in the 'svm_train_brute' function. It is
        used by the other functions to properly plot the hyperplane and to separate the data. The
        value that is returned is the minimum distance (because the margin is based off the closest
        data points, also known as the support vectors).


    4. y = svm_test_brute(w, b, x)

            This function computes the label of a data point specified by the user. If the value of
        the user's specified data point [x] is greater than 0, then its label is positive (+). If it's
        less than 0, then the label is negative (-).
    

    5. plot_data_and_boundary(data, w, b)

            This function is used to actually plot the new data clusters, which are separated by the
        hyperplane. Since we're only using 2 classes, the colors red and blue will suffice in depciting
        which clusters are which, especially if the user looks at the original plot made by the functions
        within the 'helpers.py' file.



Multi-Class Classification Write-Up:
    1. [W, B] = svm_train_multiclass(training_data)

            This function essentially creates multiple classes that will be registered under a new
        variable [C], which will be used in order to calculate the data points. Those new data points
        (labeled 'data_binary') ran through the 'svm_train_brute' function. An adjusted [W] and [B]
        will be returned for the 'svm_test_multiclass' and 'plot_data_and_boundaries' functions.


    2. y = svm_test_multiclass(W, B, x)

            This functions tests the trained data points created in the 'svm_train_multiclass'
        function and creates 'scores' for them. If These are essentially labels so the multiple
        data classes can be separated properly.


    3. plot_data_and_boundaries(data, W, B)

            This function is used to plot all the multiple classes, as well as the hyperplanes and
        margins that separate them all.