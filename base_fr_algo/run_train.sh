python3 train_test_basefr.py --config_file ./configs/mobi_BL.yaml --model_name it_mobi.pkl --situation IT --opts OUTPUT.DIR ./base_fr/mobinet/ OUTPUT.VERBOSE True
python3 train_test_basefr.py --config_file ./configs/mobi_BL.yaml --model_name accom_mobi.pkl --situation ACCOM --opts OUTPUT.DIR ./base_fr/mobinet/ OUTPUT.VERBOSE True
python3 train_test_basefr.py --config_file ./configs/mobi_BL.yaml --model_name bank_mobi.pkl --situation BANK --opts OUTPUT.DIR ./base_fr/mobinet/ OUTPUT.VERBOSE True
python3 train_test_basefr.py --config_file ./configs/mobi_BL.yaml --model_name wait_mobi.pkl --situation WAIT --opts OUTPUT.DIR ./base_fr/mobinet/ OUTPUT.VERBOSE True

python3 train_test_basefr.py --config_file ./configs/rcnn_BL.yaml --model_name it_rcnn.pkl --situation IT --opts OUTPUT.DIR ./base_fr/rcnn/ OUTPUT.VERBOSE True
python3 train_test_basefr.py --config_file ./configs/rcnn_BL.yaml --model_name accom_rcnn.pkl --situation ACCOM --opts OUTPUT.DIR ./base_fr/rcnn/ OUTPUT.VERBOSE True
python3 train_test_basefr.py --config_file ./configs/rcnn_BL.yaml --model_name bank_rcnn.pkl --situation BANK --opts OUTPUT.DIR ./base_fr/rcnn/ OUTPUT.VERBOSE True
python3 train_test_basefr.py --config_file ./configs/rcnn_BL.yaml --model_name wait_rcnn.pkl --situation WAIT --opts OUTPUT.DIR ./base_fr/rcnn/ OUTPUT.VERBOSE True