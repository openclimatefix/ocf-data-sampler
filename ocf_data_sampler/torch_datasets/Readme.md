# Torch Datasets

The aim of this folder is to create torch datasets which can easily be used in our ML models and ML deployment.

## PVNet UK Regional

This pipline is for creating GSP predictions which we have used in our PVNet model.

### Init

Starting up this up, we get
- Time and locations Pairs: A list of all valid time and locations for the data
- Data: The Data is lazily loaded in, and ready to be selected. 

```mermaid
graph TD
    A1([Load GSP])
    A2([Load NWP])
    A3([Load Satellite])
    D0([All GSP Locations])
    A1 --> D1
    A2 --> D1
    A3 --> D1
    D0 --> D2
    A1 --> D3
    A2 --> D3
    A3 --> D3
    D1[Valid Time Periods] --> D2[Time and Locations Pairs]
    D3[Data]
```
### Get a Sample

```mermaid
graph TD
    A0([Index])
    A1([Time and Locations Pairs])
    A0 --> B0
    A1 --> B0
    B0[One Time and Location]
    B1([Data])
    B0 --> D0
    B1 --> D0
    D0[Filter by Location \n GSP, Satellite and NWP] --> D1
    D1[Filter by Time \n GSP, Satellite and NWP] --> D2
    D2[Load into Memory] --> E0
    E0[Add GSP Sun Features]
    E0 --> F[Example]
```
