# Torch Datasets

The aim of this folder is to create torch datasets which can easily be used in our ML models and ML deployment that use pvnet.

## generation 

## Note on t0 time (prediction time)
t0 represents the time the forecast is made and relates to a time of one of the generation values.   
This time is important because the available NWP/weather data changes depending on t0. Also to account for live delays in data, t0 is used to assume the correct amount of data that would be available at inference time.

Note that t0 is considered as past of the history we pass to the model and the first forecasted value will be for t0 + time_resolution_minutes of the generation data (e.g. 15 mins or 30 mins). 

## PVNet Dataset

This dataset is for creating samples for PVNet for renewable energy forecasting.

### Init

Starting up this up, we get
- Time and locations Pairs: A list of all valid time and locations for the data. How this is created differs slightly depending on whether there are non overlapping periods.
- Data: The Data is lazily loaded in, and ready to be selected. 

```mermaid
graph TD
    A1([Load Generation data])
    A2([Load NWP])
    A3([Load Satellite])
    A1 --> D1
    A2 --> D1
    A3 --> D1
    A1 --> D2
    A1 --> D3
    A2 --> D3
    A3 --> D3
    D1[Valid Time Periods] --> D2[T0 and Location Pairs]
    D3[Data]
```

### Get a Sample

```mermaid
graph TD
    A0([Index])
    A1([Time and Locations Pairs])
    A0 --> B0
    A1 --> B0
    B0[T0 and Location]
    B1([Data])
    B0 --> D0
    B1 --> D0
    D0[Slice by space, <br> using location, Satellite and NWP] --> D1
    D1[Slice by time <br> using location, Satellite and NWP] --> D2
    D2[Load into Memory] --> E0
    E0[Extra processing and add features like sun/time encodings]
    E0 --> F[Sample]
```
