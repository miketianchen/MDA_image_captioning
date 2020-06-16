import sys, json
import random
import os
#
# # ORIGINAL CAPTIONS
# original_json_string = '{"rsicd_00001.jpg": "a football field with several buildings surrounded", "rsicd_airport_55.jpg": "the terminal buildings including three circular buildings tarmac and runways are built in the field of sparse grass  with one plane parked on the tarmac.", "rsicd_bareland_2.jpg": "some clusters of plants grow on the brown bare land while no plants grows on the khaki bare land.", "rsicd_baseballfield_1.jpg": "five baseball fields are surrounded by green trees .", "rsicd_beach_15.jpg": "a yellow beach is near a piece of green ocean .", "rsicd_bridge_21.jpg": "the river banks decorated with trees grass and houses are connected by a bridge.", "rsicd_center_24.jpg": "the center of the square is a ring of gray blue .", "rsicd_church_15.jpg": "many cars are near a church surrounded by several buildings .", "rsicd_commercial_1.jpg": "five tall buildings in it while with a light brown highway besides .", "rsicd_denseresidential_16.jpg": "many buildings and many green trees are in a dense residential area .", "rsicd_desert_9.jpg": "the desert for a trench .", "rsicd_farmland_10.jpg": "there are many square blocks in the farmland .", "rsicd_forest_11.jpg": "many green trees are in a piece of forest .", "rsicd_industrial_4.jpg": "there is a factory near a road decorated with some trees.", "rsicd_meadow_2.jpg": "this is a big farmland .", "rsicd_mediumresidential_11.jpg": "some buildings and green trees are in a medium residential area .", "rsicd_mountain_20.jpg": "three lakes are lying among mountains with white and green mountain peaks.", "rsicd_park_3.jpg": "the lake is surrounded by many thick trees .", "rsicd_parking_8.jpg": "the parking lot, inside there is a circular parking lot .", "rsicd_playground_2.jpg": "this is a playground .", "rsicd_pond_2.jpg": "many green trees are around an irregular pond .", "rsicd_port_1.jpg": "some boats are in a port near some buildings .", "rsicd_railwaystation_8.jpg": "the railway station traffic is very convenient .", "rsicd_resort_1.jpg": "many buildings in a resort are near an ocean .", "rsicd_river_3.jpg": "many buildings and some farmlands are near a river .", "rsicd_school_1.jpg": "many buildings and some green trees are in a school near a river .", "rsicd_sparseresidential_3.jpg": "many trees were planted around the white house .", "rsicd_square_1.jpg": "a square is near a building and some green trees .", "rsicd_stadium_2.jpg": "there is a stadium between a road and a sports field .", "rsicd_storagetanks_7.jpg": "in front of the storage tanks for a clearing .", "rsicd_viaduct_17.jpg": "many green plants are near a viaduct with some cars .", "ucm_69.jpg": "There is a piece of farmland ."}'
#
# original_captions = json.loads(original_json_string)
#
# # GENERATED CAPTIONS
# caption_json_string = '{"rsicd_00001.jpg": "many buildings and some green trees are in a commercial area .", "rsicd_airport_55.jpg": "several planes are parked in an airport with a parking lot .", "rsicd_bareland_2.jpg": "it is a piece of bare land.", "rsicd_baseballfield_1.jpg": "many green trees are around a circle square with green meadows .", "rsicd_beach_15.jpg": "yellow beach is near a piece of green ocean with a line of white wave .", "rsicd_bridge_21.jpg": "a bridge is over a river with some green trees and several buildings in two sides .", "rsicd_center_24.jpg": "a large building is near a road .", "rsicd_church_15.jpg": "a church is near a road with some cars .", "rsicd_commercial_1.jpg": "a playground is surrounded by many buildings and some green trees", "rsicd_denseresidential_16.jpg": "many buildings and green trees are in a dense residential area .", "rsicd_desert_9.jpg": "a piece of sand in the desert is like fish scale .", "rsicd_farmland_10.jpg": "many pieces of farmlands are together .", "rsicd_forest_11.jpg": "many green trees are in a forest .", "rsicd_industrial_4.jpg": "many buildings and green trees are in a school .", "rsicd_meadow_2.jpg": "it is a piece of green meadow .", "rsicd_mediumresidential_11.jpg": "many buildings are around a park with many green trees and a pond .", "rsicd_mountain_20.jpg": "it is a piece of green mountain .", "rsicd_park_3.jpg": "many green trees and some buildings are in a park .", "rsicd_parking_8.jpg": "many cars are parked in a parking lot near a road .", "rsicd_playground_2.jpg": "a playground is near several buildings and green trees .", "rsicd_pond_2.jpg": "many green trees are around an irregular pond .", "rsicd_port_1.jpg": "many boats are in a port near many buildings .", "rsicd_railwaystation_8.jpg": "many buildings are in an industrial area .", "rsicd_resort_1.jpg": "A villa with grey roofs is in the sparse residential area .", "rsicd_river_3.jpg": "many buildings and green trees are in a resort with a pond .", "rsicd_school_1.jpg": "many buildings and some green trees are in a commercial area .", "rsicd_sparseresidential_3.jpg": "many green trees are in two sides of a curved river .", "rsicd_square_1.jpg": "There are some buildings with grey roofs pressed together .", "rsicd_stadium_2.jpg": "a playground is near several buildings and green trees .", "rsicd_storagetanks_7.jpg": "a large building is near a football field .", "rsicd_viaduct_17.jpg": "many green trees and some buildings are near a viaduct .", "ucm_69.jpg": "It is a piece of cropland ."}'
#
# captions = json.loads(caption_json_string) # deserialises it
#
# # SCORE
# score_json_string = '{"rsicd_00001.jpg": {"Bleu_1": 0.7272727272066117, "Bleu_2": 0.6030226890979661, "Bleu_3": 0.43233287812182764, "Bleu_4": 5.637560314657452e-05, "METEOR": 0.38391201392497726, "ROUGE_L": 0.7128547579298832, "CIDEr": 0.476917571151693, "SPICE": 0.3448275862068965, "USC_similarity": 0.5888541102409363}, "rsicd_airport_55.jpg": {"Bleu_1": 0.6363636363057852, "Bleu_2": 0.43693144871094447, "Bleu_3": 0.34876912508999725, "Bleu_4": 0.2698553466394427, "METEOR": 0.1463852429656055, "ROUGE_L": 0.2993865030674847, "CIDEr": 0.19540625306186687, "SPICE": 0.17647058823529413, "USC_similarity": 0.43447749614715575}, "rsicd_bareland_2.jpg": {"Bleu_1": 0.41960713664321503, "Bleu_2": 0.32048024429805333, "Bleu_3": 0.21584437029713033, "Bleu_4": 3.3307187386618874e-05, "METEOR": 0.20055186998716348, "ROUGE_L": 0.5024711696869852, "CIDEr": 0.40126957924217255, "SPICE": 0.3076923076923077, "USC_similarity": 0.35291931629180906}, "rsicd_baseballfield_1.jpg": {"Bleu_1": 0.4545454545041323, "Bleu_2": 0.21320071633525958, "Bleu_3": 1.715714163827573e-06, "Bleu_4": 5.01257871520395e-09, "METEOR": 0.1812455078557206, "ROUGE_L": 0.216696269982238, "CIDEr": 0.04483035780326659, "SPICE": 0.16666666666666669, "USC_similarity": 0.4804593801498413}, "rsicd_beach_15.jpg": {"Bleu_1": 0.6666666666222223, "Bleu_2": 0.6172133998057506, "Bleu_3": 0.5897597461746652, "Bleu_4": 0.56591192562306, "METEOR": 0.5257043737612862, "ROUGE_L": 0.7469387755102042, "CIDEr": 3.547532266833339, "SPICE": 0.7368421052631579, "USC_similarity": 0.8906110525131226}, "rsicd_bridge_21.jpg": {"Bleu_1": 0.8749999998906252, "Bleu_2": 0.683130050977159, "Bleu_3": 0.46415888330123206, "Bleu_4": 5.266403877784736e-05, "METEOR": 0.4606256406206317, "ROUGE_L": 0.375, "CIDEr": 1.2654449473975247, "SPICE": 0.41860465116279066, "USC_similarity": 0.7765923976898194}, "rsicd_center_24.jpg": {"Bleu_1": 0.3715190997867868, "Bleu_2": 0.23168286400471463, "Bleu_3": 2.103416377214603e-06, "Bleu_4": 6.701444468735928e-09, "METEOR": 0.08303629239980734, "ROUGE_L": 0.2846034214618974, "CIDEr": 0.22994358895211153, "SPICE": 0.08695652173913043, "USC_similarity": 0.23240332901477814}, "rsicd_church_15.jpg": {"Bleu_1": 0.3977063629402297, "Bleu_2": 0.2982797722031009, "Bleu_3": 2.248872709912558e-06, "Bleu_4": 6.417592062678014e-09, "METEOR": 0.21565728429719505, "ROUGE_L": 0.20854700854700853, "CIDEr": 2.3681951863757407, "SPICE": 0.3636363636363636, "USC_similarity": 0.8342411518096924}, "rsicd_commercial_1.jpg": {"Bleu_1": 0.5810640920739736, "Bleu_2": 0.3257514508263493, "Bleu_3": 2.2080875908320444e-06, "Bleu_4": 5.9206501149795045e-09, "METEOR": 0.25917072525804385, "ROUGE_L": 0.34512022630834516, "CIDEr": 0.2042666190312274, "SPICE": 0.19047619047619044, "USC_similarity": 0.3966906726360321}, "rsicd_denseresidential_16.jpg": {"Bleu_1": 0.9131007161162442, "Bleu_2": 0.8662433988135071, "Bleu_3": 0.8107457798257313, "Bleu_4": 0.7426141116403059, "METEOR": 0.5202750998037486, "ROUGE_L": 0.949080622347949, "CIDEr": 7.461076038972971, "SPICE": 0.9473684210526316, "USC_similarity": 0.9939653873443604}, "rsicd_desert_9.jpg": {"Bleu_1": 0.4545454545041323, "Bleu_2": 0.21320071633525958, "Bleu_3": 1.715714163827573e-06, "Bleu_4": 5.01257871520395e-09, "METEOR": 0.20123209599592948, "ROUGE_L": 0.2681318681318681, "CIDEr": 0.17114518197229367, "SPICE": 0.11111111111111112, "USC_similarity": 0.395147317647934}, "rsicd_farmland_10.jpg": {"Bleu_1": 0.9999999996666668, "Bleu_2": 0.9999999996500001, "Bleu_3": 0.999999999627778, "Bleu_4": 0.9999999995958335, "METEOR": 1.0, "ROUGE_L": 1.0, "CIDEr": 6.014315525520365, "SPICE": 0.7142857142857143, "USC_similarity": 0.853948450088501}, "rsicd_forest_11.jpg": {"Bleu_1": 0.7514772928605784, "Bleu_2": 0.6860017745795912, "Bleu_3": 0.6564758827182055, "Bleu_4": 0.6319145616813112, "METEOR": 0.43449266989344915, "ROUGE_L": 0.8557114228456915, "CIDEr": 2.7074993730099726, "SPICE": 0.5000000000000001, "USC_similarity": 0.7438536763191224}, "rsicd_industrial_4.jpg": {"Bleu_1": 0.7777777776049386, "Bleu_2": 0.3118047821597068, "Bleu_3": 2.4037492832749525e-06, "Bleu_4": 6.9363190820961554e-09, "METEOR": 0.2298942116862121, "ROUGE_L": 0.4444444444444444, "CIDEr": 0.10626689948299414, "SPICE": 0.3076923076923077, "USC_similarity": 0.37407684326171875}, "rsicd_meadow_2.jpg": {"Bleu_1": 0.9999999997142859, "Bleu_2": 0.9999999997023811, "Bleu_3": 0.9999999996873018, "Bleu_4": 0.9999999996672622, "METEOR": 1.0, "ROUGE_L": 1.0, "CIDEr": 4.0736539495070065, "SPICE": 0.6666666666666666, "USC_similarity": 0.6968465566635131}, "rsicd_mediumresidential_11.jpg": {"Bleu_1": 0.46153846150295863, "Bleu_2": 0.1961161351224697, "Bleu_3": 1.5177887247853536e-06, "Bleu_4": 4.324227075083586e-09, "METEOR": 0.22860481521932893, "ROUGE_L": 0.33841886269070737, "CIDEr": 0.16985317718271592, "SPICE": 0.2608695652173913, "USC_similarity": 0.4490794062614441}, "rsicd_mountain_20.jpg": {"Bleu_1": 0.41960713664321503, "Bleu_2": 0.32048024429805333, "Bleu_3": 0.21584437029713033, "Bleu_4": 3.3307187386618874e-05, "METEOR": 0.1087378640776699, "ROUGE_L": 0.2695139911634757, "CIDEr": 0.5727917352804647, "SPICE": 0.22222222222222218, "USC_similarity": 0.5040748000144959}, "rsicd_park_3.jpg": {"Bleu_1": 0.7999999999200001, "Bleu_2": 0.7302967432631348, "Bleu_3": 0.5848035475770537, "Bleu_4": 0.4111336168512899, "METEOR": 0.2860106620879932, "ROUGE_L": 0.44309927360774815, "CIDEr": 0.6799081867672438, "SPICE": 0.4, "USC_similarity": 0.5753564357757568}, "rsicd_parking_8.jpg": {"Bleu_1": 0.8181818181074382, "Bleu_2": 0.8090398348786642, "Bleu_3": 0.7984819697067689, "Bleu_4": 0.7860753020680614, "METEOR": 0.467536675545793, "ROUGE_L": 0.7765205091937766, "CIDEr": 4.173503928177071, "SPICE": 0.42105263157894735, "USC_similarity": 0.8085277318954468}, "rsicd_playground_2.jpg": {"Bleu_1": 0.7954127258804594, "Bleu_2": 0.5965595444062018, "Bleu_3": 0.4497745419825114, "Bleu_4": 0.34130653535207217, "METEOR": 0.308151629775544, "ROUGE_L": 0.5570776255707762, "CIDEr": 1.8972057860000735, "SPICE": 0.42105263157894735, "USC_similarity": 0.5319998860359192}, "rsicd_pond_2.jpg": {"Bleu_1": 0.9999999997500004, "Bleu_2": 0.9999999997410718, "Bleu_3": 0.9999999997301591, "Bleu_4": 0.9999999997163694, "METEOR": 1.0, "ROUGE_L": 1.0, "CIDEr": 10.0, "SPICE": 1.0, "USC_similarity": 1.0}, "rsicd_port_1.jpg": {"Bleu_1": 0.7777777776049386, "Bleu_2": 0.6972166886186177, "Bleu_3": 0.6524779400398905, "Bleu_4": 0.6104735834296803, "METEOR": 0.45424233670678776, "ROUGE_L": 0.7777777777777778, "CIDEr": 3.6145027045519367, "SPICE": 0.6250000000000001, "USC_similarity": 0.6640085577964783}, "rsicd_railwaystation_8.jpg": {"Bleu_1": 0.5714285712653063, "Bleu_2": 0.43643578034209335, "Bleu_3": 3.36478173042779e-06, "Bleu_4": 9.878765470943703e-09, "METEOR": 0.1626109487735576, "ROUGE_L": 0.3794712286158632, "CIDEr": 0.08748372817912092, "SPICE": 0.1818181818181818, "USC_similarity": 0.3276091396808624}, "rsicd_resort_1.jpg": {"Bleu_1": 0.18181818180165296, "Bleu_2": 4.26401432670519e-09, "Bleu_3": 1.2641490044408468e-11, "Bleu_4": 7.088856801504128e-13, "METEOR": 0.04081632653061224, "ROUGE_L": 0.1018363939899833, "CIDEr": 0.01979863339679064, "SPICE": 0.0, "USC_similarity": 0.20274348556995392}, "rsicd_river_3.jpg": {"Bleu_1": 0.4166666666319445, "Bleu_2": 0.2752409412576109, "Bleu_3": 0.19640024395020875, "Bleu_4": 3.0289764015178343e-05, "METEOR": 0.20883268669285343, "ROUGE_L": 0.48878205128205127, "CIDEr": 0.4304178126669974, "SPICE": 0.26666666666666666, "USC_similarity": 0.5049126148223877}, "rsicd_school_1.jpg": {"Bleu_1": 0.6821614783011186, "Bleu_2": 0.6745393231078268, "Bleu_3": 0.6657366722129151, "Bleu_4": 0.6553925768664942, "METEOR": 0.3949689790618185, "ROUGE_L": 0.7388963660834454, "CIDEr": 1.9116025028876078, "SPICE": 0.3478260869565218, "USC_similarity": 0.4936765193939209}, "rsicd_sparseresidential_3.jpg": {"Bleu_1": 0.4545454545041323, "Bleu_2": 6.741998623988867e-09, "Bleu_3": 1.7157141638275734e-11, "Bleu_4": 8.913765520446539e-13, "METEOR": 0.14854766362613941, "ROUGE_L": 0.216696269982238, "CIDEr": 0.06628755626014392, "SPICE": 0.25, "USC_similarity": 0.31883434057235716}, "rsicd_square_1.jpg": {"Bleu_1": 0.09942659073505752, "Bleu_2": 3.3348692347964305e-09, "Bleu_3": 1.1244363549562801e-11, "Bleu_4": 6.785777427559989e-13, "METEOR": 0.020151133501259445, "ROUGE_L": 0.10427350427350426, "CIDEr": 0.06497736235715292, "SPICE": 0.22222222222222224, "USC_similarity": 0.4140326976776123}, "rsicd_stadium_2.jpg": {"Bleu_1": 0.5338249351592445, "Bleu_2": 0.23115297750913813, "Bleu_3": 1.8283683143058144e-06, "Bleu_4": 5.344197270823268e-09, "METEOR": 0.166910838167787, "ROUGE_L": 0.4026402640264026, "CIDEr": 0.0625718800888439, "SPICE": 0.13333333333333333, "USC_similarity": 0.4211876213550568}, "rsicd_storagetanks_7.jpg": {"Bleu_1": 0.22062422559099293, "Bleu_2": 0.16677623831564797, "Bleu_3": 1.5993496847456557e-06, "Bleu_4": 5.1837418805413775e-09, "METEOR": 0.09220156748172594, "ROUGE_L": 0.16486486486486487, "CIDEr": 0.03859940990865455, "SPICE": 0.0, "USC_similarity": 0.11535616964101791}, "rsicd_viaduct_17.jpg": {"Bleu_1": 0.6999999998600002, "Bleu_2": 0.5577733509080639, "Bleu_3": 0.42685972157198826, "Bleu_4": 0.32466791540375595, "METEOR": 0.33649600370013677, "ROUGE_L": 0.6, "CIDEr": 3.596095938171601, "SPICE": 0.14285714285714285, "USC_similarity": 0.8023006319999695}, "ucm_69.jpg": {"Bleu_1": 0.9999999996666668, "Bleu_2": 0.9999999996500001, "Bleu_3": 0.999999999627778, "Bleu_4": 0.9999999995958335, "METEOR": 1.0, "ROUGE_L": 1.0, "CIDEr": 5.480412547315231, "SPICE": 0.7499999999999999, "USC_similarity": 0.8402148962020874}}'
#
# scores = json.loads(score_json_string)

################################################################################

# /Users/apple/Documents/MDS_labs/DSCI_591/591_capstone_2020-mda-mds/scr/visualization/mda_mds
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# PATH TO DATA FOLDER
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), 'data')

# PATH TO ALL THE IMAGES IN THE TEST FOLDER
IMAGES_PATH = os.path.join(DATA_PATH, 'test/')

# PATH TO THE EVALUATION SCORES FILE FOR THE TEST IMAGES
TEST_IMG_SCORE_JSON = os.path.join(DATA_PATH, 'score/test_img_score.json')

# PATH TO THE INPUT CAPTIONS FOR ALL THE IMAGES IN TEST FOLDER
TEST_CAPTIONS_JSON = os.path.join(DATA_PATH, 'json/test.json')

# PATH TO GENERATED CAPTIONS FOR ALL THE IMAGES IN TEST FOLDER
GENERATED_CAPTIONS_JSON = os.path.join(DATA_PATH, 'json/test_model_caption.json')

with open(TEST_CAPTIONS_JSON) as json_file:
    captions = json.load(json_file)

with open(TEST_IMG_SCORE_JSON) as json_file:
    evaluations = json.load(json_file)

with open(GENERATED_CAPTIONS_JSON) as json_file:
    generated_captions = json.load(json_file)

demo_image_names = os.listdir(IMAGES_PATH)
def return_captions():
    random_image_one_index = random.randint(0, len(demo_image_names)-1)
    random_image_two_index = random.randint(0, len(demo_image_names)-1)

    random_image_one_name = demo_image_names[random_image_one_index]
    random_image_two_name = demo_image_names[random_image_two_index]
    # output_string_one = get_captions_n_scores(random_image_name_one)
    # output_string_two = get_captions_n_scores(random_image_name_two)

    image_one_input_captions = captions[random_image_one_name]
    image_two_input_captions = captions[random_image_two_name]

    image_one_input_sen_1 = image_one_input_captions['sentences'][0]['raw']
    image_one_input_sen_2 = image_one_input_captions['sentences'][1]['raw']
    image_one_input_sen_3 = image_one_input_captions['sentences'][2]['raw']
    image_one_input_sen_4 = image_one_input_captions['sentences'][3]['raw']
    image_one_input_sen_5 = image_one_input_captions['sentences'][4]['raw']

    image_two_input_sen_1 = image_two_input_captions['sentences'][0]['raw']
    image_two_input_sen_2 = image_two_input_captions['sentences'][1]['raw']
    image_two_input_sen_3 = image_two_input_captions['sentences'][2]['raw']
    image_two_input_sen_4 = image_two_input_captions['sentences'][3]['raw']
    image_two_input_sen_5 = image_two_input_captions['sentences'][4]['raw']

    image_one_scores = evaluations[random_image_one_name]
    image_two_scores = evaluations[random_image_two_name]


    image_one_bleu_1 = str(round(image_one_scores['Bleu_1'], 2))
    image_one_bleu_2 = str(round(image_one_scores['Bleu_2'], 2))
    image_one_bleu_3 = str(round(image_one_scores['Bleu_3'], 2))
    image_one_bleu_4 = str(round(image_one_scores['Bleu_4'], 2))
    image_one_meteor = str(round(image_one_scores['METEOR'], 2))
    image_one_rouge_l = str(round(image_one_scores['ROUGE_L'], 2))
    image_one_cider = str(round(image_one_scores['CIDEr'], 2))
    image_one_spice = str(round(image_one_scores['SPICE'], 2))
    image_one_usc = str(round(image_one_scores['USC_similarity'], 2))

    image_two_bleu_1 = str(round(image_two_scores['Bleu_1'], 2))
    image_two_bleu_2 = str(round(image_two_scores['Bleu_2'], 2))
    image_two_bleu_3 = str(round(image_two_scores['Bleu_3'], 2))
    image_two_bleu_4 = str(round(image_two_scores['Bleu_4'], 2))
    image_two_meteor = str(round(image_two_scores['METEOR'], 2))
    image_two_rouge_l = str(round(image_two_scores['ROUGE_L'], 2))
    image_two_cider = str(round(image_two_scores['CIDEr'], 2))
    image_two_spice = str(round(image_two_scores['SPICE'], 2))
    image_two_usc = str(round(image_two_scores['USC_similarity'], 2))

    image_one_model_generated_sentence = generated_captions[random_image_one_name]
    image_two_model_generated_sentence = generated_captions[random_image_two_name]


    image_one_output_list = [image_one_model_generated_sentence,
                       image_one_input_sen_1,
                       image_one_input_sen_2,
                       image_one_input_sen_3,
                       image_one_input_sen_4,
                       image_one_input_sen_5,
                       image_one_bleu_1,
                       image_one_bleu_2,
                       image_one_bleu_3,
                       image_one_bleu_4,
                       image_one_meteor,
                       image_one_rouge_l,
                       image_one_cider,
                       image_one_spice,
                       image_one_usc,
                       random_image_one_name]

    image_two_output_list = [random_image_two_name,
                       image_two_input_sen_1,
                       image_two_input_sen_2,
                       image_two_input_sen_3,
                       image_two_input_sen_4,
                       image_two_input_sen_5,
                       image_two_bleu_1,
                       image_two_bleu_2,
                       image_two_bleu_3,
                       image_two_bleu_4,
                       image_two_meteor,
                       image_two_rouge_l,
                       image_two_cider,
                       image_two_spice,
                       image_two_usc,
                       image_two_model_generated_sentence]

    image_one_output_string = "%".join(image_one_output_list)
    image_two_output_string = "%".join(image_two_output_list)

    print(image_one_output_string + "@" + image_two_output_string)

    # delimit these two output
    # print(output_string_one + "@" + output_string_two)

def get_captions_n_scores(image_name):
    input_captions = captions[image_name]
    # input_sentence = {}
    # index = 1
    # for sentence in input_captions['sentences']:
    #     input_sentence['sentence_{0}'.format(index)] = sentence['raw']
    #     index += 1
    input_sentence_1 = input_captions['sentences'][0]['raw']
    input_sentence_2 = input_captions['sentences'][1]['raw']
    input_sentence_3 = input_captions['sentences'][2]['raw']
    input_sentence_4 = input_captions['sentences'][3]['raw']
    input_sentence_5 = input_captions['sentences'][4]['raw']

    caption_scores = evaluations[image_name]

    generated_cap = generated_captions[image_name]

    output_bleu_1 = str(round(caption_scores['Bleu_1'], 2))
    output_bleu_2 = str(round(caption_scores['Bleu_2'], 2))
    output_bleu_3 = str(round(caption_scores['Bleu_3'], 2))
    output_bleu_4 = str(round(caption_scores['Bleu_4'], 2))
    output_METEOR = str(round(caption_scores['METEOR'], 2))
    output_ROUGE_L = str(round(caption_scores['ROUGE_L'], 2))
    output_CIDEr = str(round(caption_scores['CIDEr'], 2))
    output_SPICE = str(round(caption_scores['SPICE'], 2))
    output_USC = str(round(caption_scores['USC_similarity'], 2))

    output_list = [generated_cap,
                   input_sentence_1,
                   input_sentence_2,
                   input_sentence_3,
                   input_sentence_4,
                   input_sentence_5,
                   output_bleu_1,
                   output_bleu_2,
                   output_bleu_3,
                   output_bleu_4,
                   output_METEOR,
                   output_ROUGE_L,
                   output_CIDEr,
                   output_SPICE,
                   output_USC,
                   image_name]
    output_string = "%".join(output_list)

    return output_string

# def read_json():
#     # RANDOM_INDEX
#     random_index_one = random.randint(0, len(captions)-1)
#     random_index_two = random.randint(0, len(captions)-1)
#     images_list = list(scores.keys())
#
#     random_image_name_one = images_list[random_index_one]
#     random_image_name_two = images_list[random_index_two]
#
#     output_string_one = getOutputData(random_image_name_one)
#     output_string_two = getOutputData(random_image_name_two)
#
#     # delimit these two output
#     print(output_string_one + "@" + output_string_two)
#
# def getOutputData(image_name):
#     output_name = str(image_name)
#     output_scores = scores[image_name]
#     output_caption = str(captions[image_name])
#
#     output_original_caption = str(original_captions[image_name])
#
#     # OUTPUT SCORES
#     output_bleu_1 = str(round(output_scores['Bleu_1'], 2))
#     output_METEOR = str(round(output_scores['METEOR'], 2))
#     output_ROUGE_L = str(round(output_scores['ROUGE_L'], 2))
#     output_CIDEr = str(round(output_scores['CIDEr'], 2))
#     output_SPICE = str(round(output_scores['SPICE'], 2))
#     output_USC = str(round(output_scores['USC_similarity'], 2))
#     output_bleu_2 = str(round(output_scores['Bleu_2'], 2))
#     output_bleu_3 = str(round(output_scores['Bleu_3'], 2))
#     output_bleu_4 = str(round(output_scores['Bleu_4'], 2))
#
#     # use `%` as deliminter
#     output_string = output_name + "%" +output_bleu_1+"%"+output_METEOR+"%"+output_ROUGE_L+"%"+output_CIDEr+"%"+output_SPICE+"%"+output_USC+'%'+output_caption +"%"+output_original_caption+"%"+output_bleu_2+"%"+output_bleu_3+"%"+output_bleu_4
#
#     return output_string
return_captions()
