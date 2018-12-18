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

  if (r1 != 1)
    {
    }
  else
    {
      cerr << "test_Rational operator != failed" << endl;
      return -1;
    }

  string s2 = "8/7";
  Rational r2 = fromstring<Rational>(s2);

  try {
    int result = r2.normalize (4);
    cerr << "test_Rational integer normalize failed to throw exception" << endl;
    return -1;
  }
  catch (std::exception& error)
    {
      cerr << "expected exception caught" << endl;
    }

  try {
    int result = r2.normalize (8);

    if (result != 7)
    {
      cerr << "integer normalize does not return expected result" << endl;
      return -1;
    }
  }
  catch (std::exception& error)
    {
      cerr << "unexpected exception caught" << endl;
      return -1;
    }
    
  cerr << "test_Rational all tests passed" << endl;

  return 0;
}
