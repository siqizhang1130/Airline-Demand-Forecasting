# -*- coding: utf-8 -*-
# check working directory
import os
os.getcwd()


import numpy as np
import pandas as pd


def airlineForecast(trainingDataFile,validationDataFile):
    # Load data
    abTrainingData = pd.read_csv(trainingDataFile)
    abValidationData = pd.read_csv(validationDataFile)
    
    #prep datafraemes for forecast
    #adds the days_prior & 
    abTrainingData = prepDataFrame(abTrainingData)
    abValidationData = prepDataFrame(abValidationData)
    
    ## Get final demand = cum_bookings when days_prior = 0
    net_zero = abTrainingData[abTrainingData['departure_date'] == abTrainingData['booking_date']]
    net_dict = net_zero.set_index('departure_date').to_dict()['cum_bookings']
    abTrainingData['final_demand'] = abTrainingData['departure_date'].map(net_dict)
    
    #select only days prior > 0
    abTrainingData = abTrainingData[abTrainingData['days_prior'] > 0]
    
    
    ### Calculate remaining demand
    abTrainingData['remain_demand'] = abTrainingData['final_demand'] - abTrainingData['cum_bookings']
    
    ### Calculate historical booking rate for given days prior
    abTrainingData['booking_rate'] = abTrainingData['cum_bookings'] / abTrainingData['final_demand']

    #rename Training columns
    abTrainingData.rename(columns = {'departure_date': 'departure_date_T', 'booking_date': 'booking_date_T',
                                     'cum_bookings':'cum_bookings_T', 'days_prior':'days_prior_T',
                                     'day_of_week':'day_of_week_T','days_prior_category':'days_prior_category_T',
                                     'final_demand':'final_demand_T', 'remain_demand':'remain_demand_T', 
                                     'booking_rate':'booking_rate_T'}, inplace = True)
    
    #rename Validation columns
    abValidationData.rename(columns = {'departure_date': 'departure_date_V', 'booking_date': 'booking_date_V',
                                     'cum_bookings':'cum_bookings_V', 'days_prior':'days_prior_V',
                                     'day_of_week':'day_of_week_V','days_prior_category':'days_prior_category_V',
                                     'final_demand':'final_demand_V','naive_forecast':'naive_forcast_V'}, inplace = True)
    
    # train model and calulate error
    model_by_dow_dp = abTrainingData.groupby(['day_of_week_T', 'days_prior_T'], as_index = False)['remain_demand_T', 'booking_rate_T'].median()
    model_by_dow_dp.rename(columns={'remain_demand_T':'add_model_dow_dp', 'booking_rate_T': 'mul_model_dow_dp'}, inplace=True)
    
    abTrainingData = abTrainingData.merge(model_by_dow_dp, left_on = ['day_of_week_T', 'days_prior_T'], right_on = ['day_of_week_T', 'days_prior_T'])
    
    abTrainingData['add_model_dow_dp_forecast'] = abTrainingData['cum_bookings_T'] + abTrainingData['add_model_dow_dp']
    abTrainingData['mul_model_dow_dp_forecast'] = abTrainingData['cum_bookings_T'] / abTrainingData['mul_model_dow_dp']

    mergedData = abValidationData.merge(model_by_dow_dp, left_on = ['day_of_week_V', 'days_prior_V'], right_on = ['day_of_week_T', 'days_prior_T'])
    mergedData['forecast'] = mergedData['cum_bookings_V'] + mergedData['add_model_dow_dp']
    
    #calculate the MASE = forecast error / benchmark error
    forecast_error = (mergedData['final_demand_V'] - mergedData['forecast']).abs().sum()
    benchmark_error = (mergedData['final_demand_V']-mergedData['naive_forcast_V']).abs().sum()
    MASE = forecast_error/benchmark_error
    #generate the final dataframe
    forecasts = mergedData[['departure_date_V','booking_date_V','forecast']]

    
    return MASE, forecasts

#add the neccesary columns for forecasting
def prepDataFrame(tempDF):
    ## Create days prior column
    tempDF['departure_date'] = pd.to_datetime(tempDF['departure_date'], format = '%m/%d/%Y')
    tempDF['booking_date'] = pd.to_datetime(tempDF['booking_date'], format = '%m/%d/%Y')
    tempDF['days_prior'] = (tempDF['departure_date'] - tempDF['booking_date']).dt.days
     ## Create weekdays column
    tempDF['day_of_week'] = tempDF['booking_date'].dt.dayofweek
    tempDF['days_prior_category'] = tempDF['days_prior'].map(categorizeDaysPrior)
    
    return tempDF
    

def main():
    trainingDataFileName = "airline_booking_trainingData.csv"
    validationDataFileName = "airline_booking_validationData.csv"
    
    
    print(airlineForecast(trainingDataFileName, validationDataFileName))
    
    
main()