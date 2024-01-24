cmake CMakeLists.txt
cmake --build .
while ! ./FootChaos
do
  sleep 5
done
