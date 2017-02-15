from sklearn import tree, preprocessing

import csv
import sys


target_class = 'status'

def open_csv(file):
	with open(file) as f:
		reader = csv.reader(f)
		return list(reader)

def build_encoders(csv_file):
	# Encontra os possíveis valores(labels) para cada campo
	labels = {}

	for line in csv_file[1:]:
		for i in range(len(line)):
			try:
				if line[i] not in labels[csv_file[0][i]]:
					labels[csv_file[0][i]].append(line[i])
			except KeyError:
				labels[csv_file[0][i]] = []
				labels[csv_file[0][i]].append(line[i])

	# Montar os objetos responsaveis por traduzir as strings em numeros
	encoders = {}
	for item in csv_file[0]:
		encoders[item] = preprocessing.LabelEncoder()
		encoders[item].fit(labels[item])

	return encoders

def preprocess_data(csv_file, encoders):
	# Montar as listas de dados e de targets
	data = []
	target = []
	new_csv = [csv_file[0]]
	for line in csv_file[1:]:
		numerical_data = []
		encoded_csv_line = []
		for i in range(len(line)):
			col = line[i]
			encoder = encoders[csv_file[0][i]]
			encoded_data = encoder.transform([col])[0]

			if csv_file[0][i] == target_class:
				target.append(encoded_data)
			else:
				numerical_data.append(encoded_data)

			encoded_csv_line.append(encoded_data)

		data.append(numerical_data)

		new_csv.append(encoded_csv_line)

	with open('bolsa_familia_encoded.csv', 'a') as f:
		writer = csv.writer(f)
		writer.writerows(new_csv)


	return data, target

def prepare_data(csv_file):
	data = []
	target = []
	for line in csv_file[1:]:
		numerical_data = []

		for i in range(len(line)):
			col = line[i]

			if csv_file[0][i] == target_class:
				target.append(col)
			else:
				numerical_data.append(col)

		data.append(numerical_data)

	return data, target

if __name__ == '__main__':
	default_csv = open_csv('bolsa_familia.csv')
	encoders = build_encoders(default_csv)

	try:
		csv_file = open_csv('bolsa_familia_encoded.csv')
		data, target = prepare_data(csv_file)
	except FileNotFoundError:
		data, target = preprocess_data(default_csv, encoders)

	clf = tree.DecisionTreeClassifier()
	tree = clf.fit(data, target)


	regiao = 'Grande Florianópolis'
	localizacao = 'URBANA'
	serie = '8º ano'

	regiao_enc = encoders['regiao'].transform([regiao])[0]
	localizacao_enc = encoders['localizacao'].transform([localizacao])[0]
	serie_enc = encoders['serie'].transform([serie])[0]


	prediction_enc = tree.predict([[regiao_enc, localizacao_enc, serie_enc]])
	prediction = encoders['status'].inverse_transform(int(prediction_enc[0]))

	proba = tree.predict_proba([[regiao_enc, localizacao_enc, serie_enc]])[0]

	# Output
	print('\nAcurácia do classificador: ' + str(tree.score(data, target)))

	print('\nA predição retornou: ' + prediction)
	if prediction == 'Suficiente':
		print('O aluno atende aos pré requisitos do Bolsa Família.')
	else:
		print('O aluno não atende aos pré requisitos do Bolsa Família e deve ser desligado do programa.')
	
	print('\nPeso de cada variável:')
	for i in range(len(default_csv[0])-1):
		print('\t' + default_csv[0][i] + ': ' + str(tree.feature_importances_[i]))


	print('\nGrau de certeza de cada possível resposta:')
	for i in range(len(encoders['status'].classes_)):
		print(encoders['status'].classes_[i] + ': ' + str(proba[i]))