"""
 Created on Tue Mar 25 02:16:34 2020
 Author: Gupse Eyşan Toy
 Explanation: In this assignment, we will construct a decision tree to classify butterflies and birds.

"""
import matplotlib.pyplot as plt  # in order to using plot our data
import numpy as np  # in order to using create numpy arrays
import pandas as pd  # in order to using data frame


def freqWeight(data):
    """
    This method calculates the viewing frequency of each element.
    :param data: dataset
    :return: weight
    """
    arr = np.unique(data, return_counts=True)
    sumArr = np.sum(arr[1])
    weight = arr[1] / sumArr

    return weight


def freqElements(data):
    """
    This method calculates the viewing frequency of each element.
    :param data: dataset
    :return: elements
    """
    arr = np.unique(data, return_counts=True)

    return arr[0]


def entropy(data_column):
    """
    In this method, entropy calculation is made.
    :param data_column: data column inside dataset
    :return: sum of entropy
    """

    entropy = -1 * freqWeight(data_column) * np.log2(freqWeight(data_column))
    sum = np.sum(entropy)

    return sum


def infoGainFirstLevel(feature_col, target_col):
    """
    This method calculate information gain for first level
    :param feature_col: dataset second column
    :param target_col: dataset label
    :return: information gain
    """

    target_entropy = entropy(target_col)
    elements = freqElements(feature_col)
    weights = freqWeight(feature_col)
    entropies = []

    for element in elements:
        target_arr = target_col[np.argwhere(feature_col == element)]
        entropies.append(entropy(target_arr[0]))

    pointForLevel1 = pd.DataFrame(total_df.iloc[[0], [1]]).to_numpy()
    print("Level one y-axis value:", pointForLevel1[0])

    return target_entropy - np.sum(entropies * weights)


def firstLevelPoint(feature_col, target_col):
    """
    This method find first level point coordinates
    :param feature_col: dataset second column
    :param target_col: dataset label
    :return: y-axis value
    """

    target_entropy = entropy(target_col)
    elements = freqElements(feature_col)

    entropies = []
    for element in elements:
        target_arr = target_col[np.argwhere(feature_col == element)]
        entropies.append(entropy(target_arr[0]))

    print("Target entropy", target_entropy)
    pointForLevel1 = pd.DataFrame(total_df.iloc[[0], [1]]).to_numpy()
    firstLevelPoint = pointForLevel1[0]
    print("Level one y-axis value:", firstLevelPoint)

    return firstLevelPoint


def informationGainLevelTwo(feature_col, target_col):
    """
    This method calculate information gain for second level
    :param feature_col: dataset first column
    :param target_col: label
    :return: information gain for level two
    """

    target_entropy = entropy(target_col)
    elements = freqElements(feature_col)
    weights = freqWeight(feature_col)

    entropies = []
    for element in elements:
        entropies.append(entropy(target_col[np.argwhere(feature_col == element)][0]))

    return target_entropy - np.sum(entropies * weights)


def secondLevelPoint(feature_col, target_col):
    """
    This method find first level point coordinates
    :param feature_col: dataset first column
    :param target_col: label
    :return: x-axis value
    """

    elements = freqElements(feature_col)

    entropies = []
    for element in elements:
        entropies.append(entropy(target_col[np.argwhere(feature_col == element)][0]))

    levelTwoPointYCoordinate = pd.DataFrame(levelTwoDataFrame.iloc[[0], [1]]).to_numpy()
    print("Level two point y coordinate", levelTwoPointYCoordinate[0])

    return levelTwoPointYCoordinate


def informationGainLevelThree(feature_col, target_col):
    """
    This method calculate information gain for third level
    :param feature_col: dataset second column
    :param target_col: label
    :return: information gain for level three
    """

    target_entropy = entropy(target_col)
    elements = freqElements(feature_col)
    weights = freqWeight(feature_col)

    entropies = []
    for element in elements:
        entropies.append(entropy(target_col[np.argwhere(feature_col == element)][0]))

    return target_entropy - np.sum(entropies * weights)


def thirdLevelPoint(feature_col, target_col):
    """
    This method find first level point coordinates
    :param feature_col: dataset second column
    :param target_col: label
    :return: y-axis value
    """

    elements = freqElements(feature_col)

    entropies = []
    for element in elements:
        entropies.append(entropy(target_col[np.argwhere(feature_col == element)][0]))

    levelThreePointYCoordinate = pd.DataFrame(levelThreeDataFrame.iloc[[0], [1]]).to_numpy()
    print("Level two point y coordinate", levelThreePointYCoordinate[0])

    return levelThreePointYCoordinate


def plot_data(x, y, levelOnePointX, levelOnePointY, level2Point):
    """
    This function plot draws the appropriate graph for the data thus using the numpy arrays.
    This method including three parameters these are x,y and title.
    X is test, y is predictions and z is title.
    """
    plt.figure(figsize=(5, 5))

    levelTwoPoint = level2Point
    levelTwoPointX = [levelTwoPoint[0], levelTwoPoint[0]]
    levelTwoPointY = [0, 4]




    plt.plot(levelOnePointX, levelOnePointY, levelTwoPointX, levelTwoPointY)

    plt.scatter(x[:, :2][y == 1, 0], x[:, :2][y == 1, 1], c='#ffff00', s=0.90)

    plt.scatter(x[:, :2][y == 2, 0], x[:, :2][y == 2, 1], c='#00bfff', s=0.90)

    plt.text(0.1, 4.1, "Level 1: y=3.99 IG:0.26111", fontsize=7, color='#0000ff')
    plt.text(3.1, 3.7, "Level 2: x=3.013 IG:0.99453", fontsize=7, color='#f9ac26')


    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == "__main__":
    """
     This main part performs the application of knn using knn and plot_data methods and than draws the data obtained as a result. 
    """

    # initializing
    predictions = []
    x = 0
    y = 1

    # read txt file and give column names with using the pandas dataframe
    columnNames = ['x', 'y', 'label']
    data = pd.read_csv("data.txt", delimiter=',', names=columnNames, header=None)

    # create dataFrame using data firt and third column: SepalLength and PetalWidth and also using fourth column this is label.
    dataFrame = data.iloc[:, [0, 1, 2]]
    print(len(dataFrame))

    # create labels for this using the last column in the dataFrame
    labels = dataFrame.iloc[:, -1]

    # divide training sets first 30 elements for each different label
    train_data1 = dataFrame.iloc[0:69]
    train_label1 = labels.iloc[0:69]
    train_data2 = dataFrame.iloc[101:249]
    train_label2 = labels.iloc[101:249]

    # divide test sets first 30 elements for each different label
    test_data1 = dataFrame.iloc[69:101]
    test_label1 = labels.iloc[69:101]
    test_data2 = dataFrame.iloc[249:]
    test_label2 = labels.iloc[249:]

    # append the discrete training data and create to be used training data
    training = train_data1.append(train_data2).values
    trainingl = train_label1.append(train_label2).values

    # append the discrete test data and create to be used test data
    test = test_data1.append(test_data2).values
    testl = test_label1.append(test_label2).values

    # convet training and test dataframe part
    training_df = pd.DataFrame(training)
    test_df = pd.DataFrame(test)

    total_df = training_df.append(test_df)
    arr_total = np.array(total_df).reshape([-1, 1])

    dataFrameFirstColumn = np.array(total_df[0]).reshape([-1, 1])
    dataFrameSecondColumn = np.array(total_df[1]).reshape([-1, 1])
    target2 = np.array(total_df[2]).reshape([-1, 1])

    # Create numpy array using data
    data = data.iloc[:, [0, 1, 2]].values
    firstLevelPointCoordinate = firstLevelPoint(dataFrameSecondColumn, target2)
    print(firstLevelPointCoordinate)

    levelOnePointX = [0, 8]
    levelOnePointY = [firstLevelPointCoordinate + 0.001, firstLevelPointCoordinate + 0.001]

    levelOnePoint = 3.993816
    dataFrame1 = dataFrame[dataFrame['y'] < levelOnePoint]
    print(dataFrame1)
    levelTwoDataFrame = pd.DataFrame(dataFrame1)
    levelTwoArray = levelTwoDataFrame.to_numpy()
    print(levelTwoArray[:, 0])
    print(levelTwoArray[:, 1])


    findLevelTwoPoint = secondLevelPoint(levelTwoArray[:, 0], levelTwoArray[:, 2])
    print("Level two point", findLevelTwoPoint)
    print("information gain for level two", informationGainLevelTwo(levelTwoArray[:, 0], levelTwoArray[:, 2]))


    # convert labels
    for i in range(len(data)):
        if data[i][2] == '1':
            data[i][2] = 1
        elif data[i][2] == '2':
            data[i][2] = 2

    # Calculate information gain for first level

    print("Information gain for first level:", infoGainFirstLevel(dataFrameSecondColumn, target2))

    label = data[:, 2]
    title = "Original data"

    # Call plot_data method
    level2Point = secondLevelPoint(levelTwoArray[:, 0], levelTwoArray[:, 2])
    plot_data(data, label, levelOnePointX, levelOnePointY, level2Point)
