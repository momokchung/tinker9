#pragma once

namespace tinker
{
class DelcxSort {
public:
   inline void valsort2(int a, int b, int& ia, int& ib, int& iswap);
   inline void valsort3(int a, int b, int c, int& ia, int& ib, int& ic, int& iswap);
   inline void valsort4(int a, int b, int c, int d, int& ia, int& ib, int& ic, int& id, int& iswap);
   inline void valsort5(int a, int b, int c, int d, int e, int& ia, int& ib, int& ic, int& id, int& ie, int& iswap);
   inline void sort4_sign(int* list, int* idx, int& iswap, int n);
};

// valsort2: sort two integers and count number of swaps
inline void DelcxSort::valsort2(int a, int b, int& ia, int& ib, int& iswap)
{
   iswap = 1;
   if (a > b) {
      ia = b;
      ib = a;
      iswap = -iswap;
   }
   else {
      ia = a;
      ib = b;
   }
}

// valsort3: sort three integers and count number of swaps
inline void DelcxSort::valsort3(int a, int b, int c, int& ia, int& ib, int& ic, int& iswap)
{
   valsort2(a, b, ia, ib, iswap);

   ic = c;

   int temp;
   if (ib > ic) {
      temp = ib;
      ib = ic;
      ic = temp;
      iswap = -iswap;
      if (ia > ib) {
         temp = ia;
         ia = ib;
         ib = temp;
         iswap = -iswap;
      }
   }
}

// valsort4: sort four integers and count number of swaps
inline void DelcxSort::valsort4(int a, int b, int c, int d, int& ia, int& ib, int& ic, int& id, int& iswap)
{
   valsort3(a, b, c, ia, ib, ic, iswap);

   id = d;

   int temp;
   if (ic > id) {
      temp = ic;
      ic = id;
      id = temp;
      iswap = -iswap;
      if (ib > ic) {
         temp = ib;
         ib = ic;
         ic = temp;
         iswap = -iswap;
         if (ia > ib) {
            temp = ia;
            ia = ib;
            ib = temp;
            iswap = -iswap;
         }
      }
   }
}

// valsort5: sort five integers and count number of swaps
inline void DelcxSort::valsort5(int a, int b, int c, int d, int e, int& ia, int& ib, int& ic, int& id, int& ie, int& iswap)
{
   valsort4(a, b, c, d, ia, ib, ic, id, iswap);

   ie = e;
   int temp;

   if (id > ie) {
      temp = id;
      id = ie;
      ie = temp;
      iswap = -iswap;
      if (ic > id) {
         temp = ic;
         ic = id;
         id = temp;
         iswap = -iswap;
         if (ib > ic) {
            temp = ib;
            ib = ic;
            ic = temp;
            iswap = -iswap;
            if (ia > ib) {
               temp = ia;
               ia = ib;
               ib = temp;
               iswap = -iswap;
            }
         }
      }
   }
}

// sort4_sign: sorts the list of 4 numbers, and computes the signature of the permutation
inline void DelcxSort::sort4_sign(int* list, int* idx, int& iswap, int n)
{
   for (int i = 0; i < n; i++) idx[i] = i;

   iswap = 1;

   int a;

   for (int i = 0; i < n-1; i++) {
      for (int j = i+1; j < n; j++) {
         if (list[i] > list[j]) {
            a = list[i];
            list[i] = list[j];
            list[j] = a;
            a = idx[i];
            idx[i] = idx[j];
            idx[j] = a;
            iswap = -iswap;
         }
      }
   }
}
}
