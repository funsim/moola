cl__1 = 1;
r = 2.5;
x = 30;
y = 10;
c = 10;

N = 1;
Nm = N/2;
Nc = N/10;

Point(1) = {0, 0, 0, N};
Point(2) = {x, 0, 0, N};
Point(3) = {x, y/2, 0, Nm};
Point(4) = {0, y/2, 0, Nm};
Point(6) = {c, y/2, 0, Nm};
Point(5) = {c-r, y/2, 0, Nc};
Point(8) = {c, y/2-r, 0, Nc};
Point(7) = {c+r, y/2, 0, Nc};

Line(1) = {5, 4};
Line(2) = {4, 1};
Line(3) = {1, 2};
Line(4) = {2, 3};
Line(5) = {3, 7};
Circle(6) = {5, 6, 8};
Circle(7) = {8, 6, 7};
Line Loop(8) = {1, 2, 3, 4, 5, -7, -6};
Plane Surface(9) = {8};
Symmetry {0, 1, 0, -y/2} {
  Duplicata{Surface{9};} 
}
Physical Surface(18) = {10, 9};
Physical Line(1) = {12, 2};
Physical Line(2) = {13, 17, 6, 7, 16, 3};
Physical Line(3) = {4, 14};
