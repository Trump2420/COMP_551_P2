import math


def words_from_file(file):
    """this function takes a file as input and returns a list of words """
    file_handle = open(file, "r")
    content = file_handle.read()  # create a string from the file
    file_handle.close()
    lst = content.split()
    return lst


def list_from_file(file):
    """this function takes a file as input and returns a list of strings """
    file_handle = open(file, "r")
    content = file_handle.readlines()  # create a string from the file
    file_handle.close()
    data_set_list = []
    for line in content:
        temp_list = line.splitlines()
        data_set_list.append(temp_list)
    return data_set_list


def words_from_list(lst):
    """this function takes a list of strings as input and returns a list of all words inside that given list"""
    result = []
    for v in lst:
        temp_list = v[0].split()
        result.extend(temp_list)
    return result


def category_list_maker():
    """this function takes training input file and returns records of each of 8 classes as a list"""
    hockey_list = []
    movies_list = []
    nba_list = []
    news_list = []
    nfl_list = []
    politics_list = []
    soccer_list = []
    worldnews_list = []
    for index, category in enumerate(train_output_list):
        if category == "hockey":
            hockey_list.append(train_input_list[index])
        elif category == "movies":
            movies_list.append(train_input_list[index])
        elif category == "nba":
            nba_list.append(train_input_list[index])
        elif category == "news":
            news_list.append(train_input_list[index])
        elif category == "nfl":
            nfl_list.append(train_input_list[index])
        elif category == "politics":
            politics_list.append(train_input_list[index])
        elif category == "soccer":
            soccer_list.append(train_input_list[index])
        elif category == "worldnews":
            worldnews_list.append(train_input_list[index])
    return hockey_list, movies_list, nba_list, news_list, nfl_list, politics_list, soccer_list, worldnews_list


def frequency_counter(xs, ys):
    """this function takes a list of words and a dictionary as input and returns a list that corresponds to
    the frequency of each word in dictionary that appears in the list """
    xs.sort()  # this is sorted input list
    #  creating a list with the length of dictionary(ys) and fill it with 0
    counter_list = []
    for i in range(len(ys)):
        counter_list.append(0)
    xi = 0
    yi = 0

    while True:
        if xi >= len(xs):
            break
        if yi >= len(ys):
            break
        if xs[xi] == ys[yi]:
            # increment frequency of the corresponding word in dictionary
            counter_list[yi] += 1
            xi += 1
        elif xs[xi] < ys[yi]:
            xi += 1
        else:
            yi += 1
    return counter_list


def probability_calculator(frequency_list, word_list):
    """this function calculate probability of each word in dictionary given class.
    in other word P(word_k|class) = (n_k + 1)/ (n + len(dictionary)) in which, k: {words in dictionary},
    n_k: number of time word k in dictionary appears in class' word set, n: number of words in class' word set"""
    result = []
    for i,word in enumerate(dictionary):
        p_word_given_class = (frequency_list[i] + 1)/(len(word_list) + len(dictionary))
        result.append(p_word_given_class)
    return result


def joint_probability(lst1, lst2):
    # print(lst1)
    # print(lst2)
    result = 0
    # i = 0
    for i, v in enumerate(lst2):
        result += math.log(v) * int(lst1[i])
        i += 1
    return result


def prediction(lst):
    category_list = ["hockey", "movies", "nba", "news", "nfl", "politics", "soccer", "worldnews"]
    probability_list = []
    class_conditional_probability_list = [p_words_given_hockey, p_words_given_movies, p_words_given_nba,
                                          p_words_given_news, p_words_given_nfl, p_words_given_politics,
                                          p_words_given_soccer, p_words_given_worldnews]
    class_probability_list = [p_hockey, p_movies, p_nba, p_news, p_nfl, p_politics, p_soccer, p_worldnews]
    temp_list = lst[0].split()
    feature_vector = frequency_counter(temp_list, dictionary)
    # print(feature_vector)
    for i, v in enumerate(class_conditional_probability_list):
        probability_list.append(joint_probability(feature_vector, v) +
                                math.log(class_probability_list[i]))
    index = probability_list.index(max(probability_list))
    return category_list[index]


def file_maker(name, lst):
    import csv
    with open(name, 'w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        file_writer.writerow(['id', 'category'])
        for i, row in enumerate(lst):
            file_writer.writerow([i, row])


# creating variables and assigning appropriate values to them
train_output_list = words_from_file("train_output_processed.csv")
train_input_list = list_from_file("train_input_processed.csv")
dictionary = words_from_file("dictionary_10000.csv")
test_input_list = list_from_file("test_input_processed.csv")
# print(test_input_list[0])

# creating a list of records for each class
hockey_list, movies_list, nba_list, news_list, nfl_list, politics_list, soccer_list, worldnews_list \
    = category_list_maker()

# finding the probability of each category(class)
p_hockey = len(hockey_list)/len(train_output_list)
p_movies = len(movies_list)/len(train_output_list)
p_nba = len(nba_list)/len(train_output_list)
p_news = len(news_list)/len(train_output_list)
p_nfl = len(nfl_list)/len(train_output_list)
p_politics = len(politics_list)/len(train_output_list)
p_soccer = len(soccer_list)/len(train_output_list)
p_worldnews = len(worldnews_list)/len(train_output_list)

# finding all words for each category and putting them into a single list of words
hockey_words_list = words_from_list(hockey_list)
movies_words_list = words_from_list(movies_list)
nba_words_list = words_from_list(nba_list)
news_words_list = words_from_list(news_list)
nfl_words_list = words_from_list(nfl_list)
politics_words_list = words_from_list(politics_list)
soccer_words_list = words_from_list(soccer_list)
worldnews_words_list = words_from_list(worldnews_list)

# finding the frequency of each word in dictionary based on each category
hockey_words_frequency = frequency_counter(hockey_words_list, dictionary)
movies_words_frequency = frequency_counter(movies_words_list, dictionary)
nba_words_frequency = frequency_counter(nba_words_list, dictionary)
news_words_frequency = frequency_counter(news_words_list, dictionary)
nfl_words_frequency = frequency_counter(nfl_words_list, dictionary)
politics_words_frequency = frequency_counter(politics_words_list, dictionary)
soccer_words_frequency = frequency_counter(soccer_words_list, dictionary)
worldnews_words_frequency = frequency_counter(worldnews_words_list, dictionary)

# finding the probability of each word in dictionary given each category P(word_k|category)
p_words_given_hockey = probability_calculator(hockey_words_frequency, hockey_words_list)
p_words_given_movies = probability_calculator(movies_words_frequency, movies_words_list)
p_words_given_nba = probability_calculator(nba_words_frequency, nba_words_list)
p_words_given_news = probability_calculator(news_words_frequency, news_words_list)
p_words_given_nfl = probability_calculator(nfl_words_frequency, nfl_words_list)
p_words_given_politics = probability_calculator(politics_words_frequency, politics_words_list)
p_words_given_soccer = probability_calculator(soccer_words_frequency, soccer_words_list)
p_words_given_worldnews = probability_calculator(worldnews_words_frequency, worldnews_words_list)

# creating the position-list for test-input
temp = []
for i, v in enumerate(test_input_list):
    temp.append(prediction(v))
    print(i)
file_maker("test_predict.csv", temp)

