#include <iostream>
#include "StepIterator_test.h"

int main(int argc, char* argv[])
{
	if(!StepIterator_test()) {
		std::cerr << "StepIterator test failed." << std::endl;
		return 1;
	}

	std::cout << "All tests passed." << std::endl;
	return 0;
}