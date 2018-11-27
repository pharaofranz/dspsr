//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Rational_h
#define __Rational_h

#include <iostream>

//! Represents a rational number
class Rational 
{
  friend std::istream& operator >> (std::istream& in, Rational& r);
  friend std::ostream& operator << (std::ostream& out, const Rational& r);

public:
  Rational(int numerator = 0, int denominator = 1); 
  double doubleValue( ) const;
  
private:
  int numerator;
  int denominator;
  void reduce( );
  
};

#endif // !defined(__Rational_h)
