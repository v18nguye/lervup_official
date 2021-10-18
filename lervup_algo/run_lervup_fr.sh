python train_test_lervup.py --config_file ./configs/mobi_rf_kmeans.yaml --model_name it_mobi.pkl --situation IT --N -1 --fr 1 --opts OUTPUT.DIR ./lervup/fr/
python train_test_lervup.py --config_file ./configs/mobi_rf_kmeans.yaml --model_name accom_mobi.pkl --situation ACCOM --N -1 --fr 1 --opts OUTPUT.DIR ./lervup/fr/
python train_test_lervup.py --config_file ./configs/mobi_rf_kmeans.yaml --model_name bank_mobi.pkl --situation BANK --N -1 --fr 1 --opts OUTPUT.DIR ./lervup/fr/
python train_test_lervup.py --config_file ./configs/mobi_rf_kmeans.yaml --model_name wait_mobi.pkl --situation WAIT --N -1 --fr 1 --opts OUTPUT.DIR ./lervup/fr/

python train_test_lervup.py --config_file ./configs/rcnn_rf_kmeans.yaml --model_name it_rcnn.pkl --situation IT --N -1 --fr 1 --opts OUTPUT.DIR ./lervup/fr/
python train_test_lervup.py --config_file ./configs/rcnn_rf_kmeans.yaml --model_name accom_rcnn.pkl --situation ACCOM --N -1 --fr 1 --opts OUTPUT.DIR ./lervup/fr/
python train_test_lervup.py --config_file ./configs/rcnn_rf_kmeans.yaml --model_name bank_rcnn.pkl --situation BANK --N -1 --fr 1 --opts OUTPUT.DIR ./lervup/fr/
python train_test_lervup.py --config_file ./configs/rcnn_rf_kmeans.yaml --model_name wait_rcnn.pkl --situation WAIT --N -1 --fr 1 --opts OUTPUT.DIR ./lervup/fr/