cmake CMakeLists.txt
cmake --build .
while ! ./FootChaosExe
do
  sleep 1
done
