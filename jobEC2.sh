/opt/anaconda3/bin/python3 /home/arman/ODT-continuous/experiments.py --dataset frog --minhypo 500 --maxhypo 2500 --hypostep 500 --criterion EC2 --metric fscore --numrands 3
/opt/anaconda3/bin/python3 /home/arman/ODT-continuous/experiments.py --dataset frog --minhypo 500 --maxhypo 2500 --hypostep 500 --criterion IG --metric fscore --numrands 3
/opt/anaconda3/bin/python3 /home/arman/ODT-continuous/experiments.py --dataset frog --minhypo 500 --maxhypo 2500 --hypostep 500 --criterion US --metric fscore --numrands 3
/opt/anaconda3/bin/python3 /home/arman/ODT-continuous/experiments.py --dataset frog --metric fscore --numrands 3 --alg efdt

/opt/anaconda3/bin/python3 /home/arman/ODT-continuous/experiments.py --dataset breastcancer --minhypo 500 --maxhypo 2500 --hypostep 500 --criterion EC2 --metric fscore --numrands 3
/opt/anaconda3/bin/python3 /home/arman/ODT-continuous/experiments.py --dataset breastcancer --minhypo 500 --maxhypo 2500 --hypostep 500 --criterion IG --metric fscore --numrands 3
/opt/anaconda3/bin/python3 /home/arman/ODT-continuous/experiments.py --dataset breastcancer --minhypo 500 --maxhypo 2500 --hypostep 500 --criterion US --metric fscore --numrands 3
/opt/anaconda3/bin/python3 /home/arman/ODT-continuous/experiments.py --dataset breastcancer --metric fscore --numrands 3 --alg efdt