import json


def convert(json_path: str, output_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)

    k_dict = {}

    for row in data:
        if bool(row['Skipped']):
            continue
        
        image_id = row['External ID']

        keypoints = []
        if len(row['Label']['objects']) > 0:
            k_list = row['Label']['objects']
            for name in ['center', 'upper', 'right', 'left']:
                for k in k_list:
                    if k['value'] == name:
                        keypoints.append([k['point']['x'], k['point']['y']])
            
        if row['Label']['classifications'][0]['answer']['value'] == 'no':
            keypoints_visible = False
        else:
            keypoints_visible = True

        k_dict[image_id] = keypoints

    with open(output_path, 'w') as f:
        json.dump(k_dict, f)


if __name__ == '__main__':
    json_path = './export-2022-03-24T13 16 39.947Z.json'
    output_path = './keypoints.json'

    convert(json_path, output_path)
