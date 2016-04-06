mkdir -p results
if ! [ -e settings.txt ]
then
	echo "no settings.txt file found";
	exit;
fi

for learnRate in 0.00001
do
sed -i '$ d' config.cfg && echo "learning_rate = $learnRate" >> config.cfg
while read p; do
	echo "processing $p `date`"
	read -r -a array <<< "$p"
	> network.jsn
	echo "{
    \"layers\": [
        {
            \"size\": 75,
            \"name\": \"input\",
            \"type\": \"input\"
        }," >> network.jsn
	for layerN in `seq 1 ${array[0]}`
	do
		echo "	{
            \"size\": ${array[$layerN]},
            \"name\": \"lstm$layerN\",
            \"bias\": 1.0,
            \"type\": \"lstm\"
        }," >> network.jsn
	done
	echo "	{
            \"size\": 39,
            \"name\": \"output\",
            \"bias\": 1.0,
            \"type\": \"softmax\"
        },
        {
            \"size\": 39,
            \"name\": \"postoutput\",
            \"type\": \"multiclass_classification\"
        }
    ]
}"  >> network.jsn

../../../build/currennt --options_file config.cfg 2>&1 > results/res${p// /_}learnRate$learnRate.txt
done <settings.txt
done
