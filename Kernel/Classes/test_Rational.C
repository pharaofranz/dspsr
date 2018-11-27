/***************************************************************************
 *
 *   Copyright (C) 2018 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "Rational.h"
#include "tostring.h"

using namespace std;

int main ()
{
  string s1 = "1/4";
  Rational r1 = fromstring<Rational>(s1);

  if (r1.doubleValue() != 0.25)
  {
    cerr << "test_Rational fail result = " << r1.doubleValue() << " != 0.25"
	 << endl;
    return -1;
  }

  cerr << "test_Rational test passed" << endl;

  return 0;
}
