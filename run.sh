cmake CMakeLists.txt
cmake --build .
while ! ./FootChaos
do
  sleep 1
done
