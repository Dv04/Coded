#include <stdio.h>
#include <avr/io.h>
#include <util/delay.h>

int main(void)
{
    DDRB = 0xFF; // Set all pins of PORTB as output
    while(1)
    {
        PORTB = 0xFF; // Turn on all LEDs
        _delay_ms(1000); // Wait for 1 second
        PORTB = 0x00; // Turn off all LEDs
        _delay_ms(1000); // Wait for 1 second
    }
    return 0;
}

