mkdir -p data/esco 
cd data/esco
git clone https://github.com/jensjorisdecorte/Skill-Extraction-benchmark.git
cd Skill-Extraction-benchmark
git checkout 157da05a24e6ecfee82e4b5d01cba68a2ed0552f
mv * ..
cd ../../..
rm -rf data/esco/Skill-Extraction-benchmark/