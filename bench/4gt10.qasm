OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
t q[11];
t q[15];
x q[16];
t q[16];
cx q[15],q[16];
t q[17];
cx q[16],q[17];
cx q[17],q[16];
cx q[16],q[17];
cx q[16],q[11];
h q[21];
t q[21];
cx q[21],q[16];
tdg q[16];
cx q[16],q[21];
cx q[21],q[16];
cx q[16],q[21];
cx q[11],q[16];
t q[16];
cx q[11],q[16];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[21];
tdg q[16];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[11];
tdg q[21];
cx q[16],q[21];
cx q[11],q[16];
h q[16];
h q[16];
t q[16];
cx q[21],q[16];
cx q[16],q[21];
cx q[21],q[16];
cx q[16],q[11];
t q[11];
h q[16];
t q[16];
cx q[16],q[15];
tdg q[15];
cx q[17],q[16];
t q[16];
cx q[17],q[16];
cx q[16],q[17];
cx q[17],q[16];
cx q[16],q[15];
tdg q[15];
tdg q[16];
cx q[16],q[17];
cx q[17],q[16];
cx q[16],q[17];
cx q[16],q[15];
cx q[17],q[16];
h q[16];
t q[16];
cx q[16],q[11];
cx q[21],q[16];
tdg q[16];
cx q[15],q[16];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[17];
t q[16];
t q[17];
cx q[16],q[17];
cx q[16],q[21];
cx q[21],q[16];
cx q[16],q[21];
cx q[11],q[16];
t q[16];
cx q[11],q[16];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[15];
tdg q[15];
tdg q[16];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[10],q[15];
cx q[11],q[16];
cx q[15],q[10];
cx q[10],q[15];
cx q[10],q[11];
h q[11];
t q[11];
cx q[15],q[16];
cx q[16],q[15];
cx q[15],q[16];
cx q[15],q[10];
h q[15];
t q[15];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[16],q[21];
cx q[17],q[16];
t q[16];
tdg q[21];
cx q[17],q[22];
cx q[22],q[17];
cx q[17],q[22];
cx q[22],q[21];
tdg q[21];
cx q[16],q[21];
cx q[21],q[16];
cx q[16],q[21];
cx q[21],q[16];
tdg q[22];
cx q[22],q[21];
h q[21];
t q[21];
cx q[16],q[21];
cx q[21],q[16];
cx q[16],q[21];
cx q[21],q[22];
t q[21];
cx q[21],q[16];
cx q[16],q[21];
cx q[21],q[16];
cx q[16],q[11];
cx q[10],q[11];
cx q[11],q[10];
cx q[10],q[11];
cx q[22],q[17];
cx q[17],q[22];
cx q[22],q[17];
cx q[12],q[17];
cx q[17],q[12];
cx q[12],q[17];
cx q[11],q[12];
h q[11];
t q[11];
t q[12];
cx q[17],q[12];
cx q[12],q[17];
cx q[17],q[12];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[17],q[22];
cx q[16],q[17];
cx q[17],q[16];
cx q[16],q[17];
cx q[11],q[16];
cx q[11],q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[12],q[17];
tdg q[16];
cx q[17],q[12];
cx q[12],q[17];
cx q[22],q[17];
t q[17];
cx q[17],q[16];
cx q[16],q[17];
cx q[17],q[16];
cx q[22],q[17];
tdg q[17];
cx q[16],q[17];
cx q[16],q[21];
cx q[21],q[16];
cx q[16],q[21];
tdg q[22];
cx q[22],q[21];
cx q[17],q[22];
h q[17];
t q[17];
cx q[17],q[12];
tdg q[12];
cx q[12],q[17];
cx q[17],q[12];
cx q[12],q[17];
cx q[11],q[12];
t q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[12],q[17];
tdg q[12];
tdg q[17];
cx q[17],q[16];
cx q[16],q[17];
cx q[17],q[16];
cx q[11],q[16];
cx q[12],q[11];
h q[11];
t q[11];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[11];
cx q[11],q[12];
t q[11];
t q[12];
cx q[11],q[12];
h q[21];
h q[21];
t q[21];
t q[22];
cx q[22],q[17];
cx q[17],q[22];
cx q[22],q[17];
cx q[16],q[17];
cx q[17],q[22];
cx q[21],q[16];
tdg q[16];
cx q[22],q[17];
cx q[17],q[22];
cx q[22],q[21];
t q[21];
cx q[22],q[17];
cx q[17],q[22];
cx q[22],q[17];
cx q[17],q[16];
tdg q[16];
tdg q[17];
cx q[17],q[22];
cx q[21],q[16];
cx q[22],q[17];
cx q[17],q[22];
cx q[16],q[17];
cx q[17],q[16];
cx q[16],q[17];
cx q[11],q[16];
cx q[16],q[11];
cx q[11],q[16];
cx q[22],q[21];
cx q[17],q[22];
h q[17];
t q[17];
cx q[17],q[16];
cx q[12],q[17];
tdg q[16];
t q[17];
cx q[17],q[12];
cx q[12],q[17];
cx q[17],q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[17],q[16];
tdg q[16];
cx q[11],q[16];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[11];
tdg q[17];
cx q[17],q[16];
h q[16];
t q[16];
cx q[17],q[12];
cx q[12],q[17];
cx q[17],q[12];
cx q[11],q[12];
cx q[12],q[11];
h q[21];
cx q[21],q[16];
cx q[16],q[21];
cx q[21],q[16];
cx q[11],q[16];
h q[11];
t q[11];
t q[16];
cx q[21],q[16];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[11];
cx q[16],q[21];
cx q[11],q[16];
t q[16];
cx q[11],q[16];
cx q[16],q[11];
cx q[11],q[16];
tdg q[21];
cx q[16],q[21];
tdg q[16];
tdg q[21];
cx q[16],q[21];
cx q[21],q[16];
cx q[16],q[21];
cx q[11],q[16];
cx q[21],q[16];
cx q[16],q[21];
cx q[21],q[16];
cx q[16],q[11];
h q[11];
cx q[21],q[16];