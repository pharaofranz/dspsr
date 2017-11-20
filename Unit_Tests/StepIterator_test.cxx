#include "StepIterator_test.h"
#include "../Kernel/Classes/dsp/StepIterator.h"
#include <iostream>

#define CHECK_TESTVAL(b, t) if(!b) std::cerr << t << std::endl;

bool StepIterator_test()
{
	int data[] = { 4, 3, 2, 1 };
	StepIterator<int> iter(data); // Test constructor
	auto iter2(iter); // Test copy-constructor
	auto iter3 = iter2; // Test assignment operator

	bool ptrWorks = (data == iter.ptr());
	CHECK_TESTVAL(ptrWorks, "ptr() function failed test.")
	bool derefWorks = (*iter == 4);
	CHECK_TESTVAL(derefWorks, "dereference operator failed test.")
	++iter;
	bool preincWorks = (*iter == 3);
	CHECK_TESTVAL(preincWorks, "pre-increment operator failed test.")
	iter.set_increment(2);
	++iter;
	++iter3;
	bool setincWorks = (*iter == 1);
	CHECK_TESTVAL(setincWorks, "set_increment() function failed test.")
	bool assignWorks = (*iter3 == 3);
	CHECK_TESTVAL(assignWorks, "assignment operator failed test.")

	return ptrWorks && derefWorks && preincWorks && setincWorks && assignWorks;
}