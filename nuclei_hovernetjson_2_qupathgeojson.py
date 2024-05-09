import json
import glob
json_files = glob.glob('/Volumes/data/UMich/nuclei/json/*.json')
# Open the JSON file
for json_file in json_files:
    geojson_file = json_file.replace('/json/','/geojson/')
    print(json_file)
    print(geojson_file)
    with open(json_file) as f:
        # Load the JSON data from the file
        data = json.load(f)

    # Access the data
    # print(data)
    geojson_features = []
    cells = data['nuc']
    # print(len(cells))
    for x in cells:
        # print(x)
        info = cells[x]
        contour = info['contour']
        contour.append(contour[0])

        geojson_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [contour]
            },
            "properties": {
            }
        }

    # Append GeoJSON feature to list
        geojson_features.append(geojson_feature)

    # Create GeoJSON object with all features
    geojson_obj = {
    "type": "FeatureCollection",
    "features": geojson_features
    }

    # Save combined GeoJSON object to file
    with open(geojson_file,'w') as f:
        json.dump(geojson_obj, f)
