# What is feature scaling?

Feature Scaling is a technique to standardize the independent features present in the data. It is performed during the data pre-processing to handle highly varying values. If feature scaling is not done then machine learning algorithm tends to use greater values as higher and consider smaller values as lower regardless of the unit of the values.

Feature scaling is a preprocessing technique that transforms feature values to a similar scale, ensuring all features contribute equally to the model.

They help improve model performance, enhance convergence and reduce biases. 

Machine learning algorithms like linear regression, logistic regression, neural network, PCA (principal component analysis), etc., that use gradient descent as an optimization technique require data to be scaled.To ensure that the gradient descent moves smoothly towards the minima and that the steps for gradient descent are updated at the same rate for all the features, we scale the data before feeding it to the model.

Distance algorithms like KNN, K-means clustering, and SVM(support vector machines) are most affected by the range of features. This is because, behind the scenes, they are using distances between data points to determine their similarity.

Tree-based algorithms, on the other hand, are fairly insensitive to the scale of the features.

# Different techniques which are used to perform feature scaling :

1. Standardization : When there are outliers in variable/data

   
   - This method of scaling is basically based on the central tendencies and variance of the data.
   - Standardization can be helpful in cases where the data follows a Gaussian distribution. 
   - Transforms data to have a mean of 0 and a standard deviation of 1 (also known as z-score scaling ie. -3 to 3 range).
   - Formula - Xscaled = (Xi - Xmean)/standard deviation.
   - Less sensitive to outliers than normalization.
   - It is useful when the feature distribution is Normal or Gaussian.
   - It is a often called as Z-Score Normalization. 

3. Normalization : When there are no ouliers in variable/data

   - This scales the range to [0, 1] or sometimes [-1, 1].
   - Formula : X_new = (X - X_mean)/(X_max - X_min)
   - Normalization is useful when there are no outliers as it cannot cope up with them.
   - It is useful when we don’t know about the distribution
   - It is a often called as Scaling Normalization

4. MinMaxcsaler :

   - Formula  : X_new = (X-X_min)/(X_max - X_min)
   - the data will range after scaling between 0 to 1.

5. RobustScaler :

   - In this method of scaling, we use two main statistical measures of the data : Median and Inter-Quartile Range.
   -  Formula : X_new = (X-Xmedian) / IQR

# Image scaling
1. Normalization : by normalization, mean dividing pixel values by 255.

   datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

2. Standardization : Standardization means subtracting the mean value of pixels and then dividing by standard deviation.

   datagen = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

3. Using opencv for scaling images :

   import cv2

   bigger = cv2.resize(image, (1050, 1610))

   **Rescaling Pixel Values:**

    import cv2
    Img = cv2.imread ('image.jpg')
    normalized = img / 255.0

4. Histogram Equalization: This spreads out pixel intensities over the whole range to improve contrast. This works well for images with low contrast where pixel values are concentrated in a narrow range. It can be applied with OpenCV using:

   eq_img = cv2.equalizeHist(img)

5. Normalizing : It means to have zero mean and unit variance. This will center the image around zero with a standard deviation of 1.This can be done by subtracting the mean and scaling to unit variance:

    mean, std = cv2.meanStdDev (img)
    std_img = (img - mean) / std

6. Some popular noise reduction techniques include:
    - Gaussian blur — Uses a Gaussian filter to blur the image and reduce high frequency noise.
    - Median blur — Replaces each pixel with the median of neighboring pixels. Effective at removing salt and pepper noise.
    - Bilateral filter — Blurs images while preserving edges. It can remove noise while retaining sharp edges.
