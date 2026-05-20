for i in {0..50..5}; do
    python main.py --ptb-type 'char' --ptb-pct $i
done

for i in {5..50..5}; do
    python main.py --ptb-type 'token' --ptb-pct $i
done

for i in {5..50..5}; do
    python main.py --ptb-type 'shuffle' --ptb-pct $i
done

for i in {5..50..5}; do
    python main.py --ptb-type 'noise' --ptb-pct $i
done