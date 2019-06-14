// ===  rect -> circular horn antenna
// ===  J.Beale August 23 2018

e1 = 0.2;       // tolerance for ID shrink
OD1 = 75.4+e1;	// OD of "3-inch" round retroreflector
RFZ2 = 4.8;    // full z-height of round reflector
RFZ = 3.1;     // z-height step of round ref. edge
RFWT = 1;     // front lip wall thickness
ODFL = OD1 + (RFWT*2);  // OD of front lip
RFZE = 2;      // extra Z overlap with cone

e = 0.4;        // extra ID for slop
TXD = 25.2+e;	// Transmitter PCB X dim
TYD = 25.2+e;   // Transmitter PCB Y DIM
// OA1 = 36;     // opening angle of horn in degrees
OA1 = 55;     // opening angle of horn in degrees
TH1 = 0.8;    // wall thickness of horn

PCBZ = 6.8;     // z-height of PCB mounting
phX = 10;       // length of hole for PCB pins
phY = 4;        // width of hold for pins
phZ = 50;       // height of cutout block for pins
phYT = -10.4;   // Y axis translation of hole from center

PI = 3.14159265358979;  // constant
SQR2 = sqrt(2);  // constant
OD2 = TXD * SQR2;  // enclosing circle for PCB

OA1R = OA1 * PI / 180;  // angle in radians
ZH0 = (OD1/2) / tan(OA1/2);
ZHd = (OD2/2) / tan(OA1/2);
Zd = 3;  // vertical shift to create wall thickness
bt = 0.6; // bottom wall thickness
// =========================================
fn = 100;	// number of facets on cylinder
eps = 0.1; // a small number

// block to remove hole for PCB pins (3x on 0.1" centers)
module pcbHole() {
    translate([0,phYT,0])
      cube([phX,phY,phZ],center=true);
}

module cone1() {
    // front lip to hold edge of round cover
 difference() {
  translate([0,0,ZH0-ZHd-Zd-RFZE]) cylinder(d = ODFL, h = RFZ+RFZE, $fn=fn);   
  translate([0,0,ZH0-ZHd-Zd-eps]) cylinder(d = OD1, h = RFZ+2*eps, $fn=fn);   
 }

  translate([0,0,-ZHd-Zd+3.5]) {
    cylinder(d1 = 0, d2=ODFL, h=ZH0-5, center=false,$fn=fn); // outer conical shell
  }
}

module cone2() {
  translate([0,0,-ZHd]) {
    cylinder(d1 = 0, d2=OD1, h=ZH0, center=false,$fn=fn); // inner conical shell
  }
}

module shell1() {
    difference() {
        cone1();
        cone2();
    }
}

module bottom1() {
  translate([0,0,-PCBZ/2]) {
    cylinder(d=TXD*SQR2+Zd, h=PCBZ, center=true,$fn=fn); 
      // bottom pcb holder side  
  }
    
}

// ========================================

// round cover in front
// color("red") { translate([0,0,5+ZH0-ZHd-Zd]) cylinder(d = OD1, h = RFZ, $fn=fn);    }



module assy1() {


// main cone
intersection() {
  translate([0,0,+OD1/2]) cube([ODFL,ODFL,OD1],center=true);
  shell1();
 }

// bottom PCB holder    
difference() {
  bottom1();  // enclosing cylinder
  translate([0,0,bt-PCBZ/2]) cube([TXD,TYD,PCBZ],center=true); // pocket
  pcbHole();  // hole for pins
}
 
} // assy1()

intersection() {
  assy1();
  // translate([0,-100,-100])  cube([200,200,200]);
    
}