# Random User Generator Documentation: https://randomuser.me/documentation
# import randomuser module
from randomuser import RandomUser
import pandas as pd

# create a random user object
r = RandomUser()

# generate 10 random users
random_user_list = r.generate_users(10)

# print the list
print(random_user_list)

# print the full_name and email for each user in the list
for user in random_user_list:
    print(user.get_full_name(), user.get_email())

# print the photo url for 5 users in the list
for user in random_user_list[:5]:
    print(user.get_picture())

# print a horizon line
print("-"*50)

# create a function that will create 10 random users, and return a dataframe with Name, Gender, City, State, Email, DOB, and Picture
def create_random_user_df():
    # create a random user object
    r = RandomUser()

    # generate 10 random users
    random_user_list = r.generate_users(10)

    # create an empty list
    random_user_df_list = []

    # iterate through the random user list
    for user in random_user_list:
        # create a dictionary of the user data
        random_user_df_list.append({"Name": user.get_full_name(),"Gender": user.get_gender(), "City": user.get_city(), "State": user.get_state(), "Email": user.get_email(), "DOB": user.get_dob(), "Picture": user.get_picture()})

    return pd.DataFrame(random_user_df_list)

# call create_random_user_df() and print the dataframe
print(create_random_user_df())

    




    


