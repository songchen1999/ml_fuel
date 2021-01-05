# ml_fuel
a tutorial ML project predicting the fuel efficency 

```
// try this out in jupyter note, you can replace with different data
url = "https://ml-fuel.herokuapp.com/"

r = requests.post(url, json ={
    'Cylinders': [4, 6, 8],
    'Displacement': [155.0, 160.0, 165.5],
    'Horsepower': [93.0, 130.0, 98.0],
    'Weight': [2500.0, 3150.0, 2600.0],
    'Acceleration': [15.0, 14.0, 16.0],
    'Model Year': [81, 80, 78],
    'Origin': [3, 2, 1]
})

r.text.strip()
```
