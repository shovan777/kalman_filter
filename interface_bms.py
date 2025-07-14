import serial
from sos import calculate_sos

ser = serial.Serial('/dev/ttyUSB0', baudrate=9600,
                    parity=serial.PARITY_NONE, bytesize=serial.EIGHTBITS, timeout=1)
print("BMS serial port opened successfully...")

def calculate_checksum(data):
    """
    Calculate the checksum for the given data.
    The checksum is the sum of all bytes in the data, modulo 256.
    """
    return sum(data) % 256


codes = {
    # find soc for daly bms
    # meaning of each of the 13 bytes:
    # start byte, host address, command id, data length, data, checksum
    'soc': b'\xA5\x40\x90\x08\x00\x00\x00\x00\x00\x00\x00\x00', #x7d,
    'temp': b'\xA5\x40\x92\x08\x00\x00\x00\x00\x00\x00\x00\x00',
}

print("Calculating checksum for each command...")
# attach checksum to each command
for key in codes:
    checksum = calculate_checksum(codes[key])
    codes[key] = codes[key] + bytes([checksum])
    # print('-----------------')
print("Checksum calculated completed successfully!!!!")



def get_cell_data(code):
    # print('inside get cell with code:', code)
    # print('fire:', codes[code])
    try:
        ser.write(codes[code])
        data = ser.readline()
        # sample soc data
        # data = b'\xa5\x01\x90\x08\x00\xc0\x00\x00u0\x03\xe8\x8e'
        # sample temp data
        # data = b'\xa5\x01\x92\x08>\x01>\x01\xff\xff\xff\xff\xba'

        # print(f'***DATA:, {data}')
        # print('length of data: ',len(data))
    except:
        print('invalid code')

    if code == 'per_cell_voltage':
        print('inside per_Cell')
        voltages_ordered = []
        if data:
            print("inside")
            for a in range(4, len(data), 2):
                # print(data[a])
                if (a+1) >= len(data)-2:
                    break
                int_data = (data[a] << 8) | data[a+1]
                float_data = int_data/1000
                print(float_data)
                voltages_ordered.append(float_data)
        print("----------------")
    elif code == 'battery_state':
        print('inside battery state')
        state = {}
        if data:
            for i in range(4, len(data), 2):
                int_data = (data[a] << 8) | data[a+1]
                print(int_data)
        print("----------------")
    elif code == 'soc':
        if data:
            # data is in 8 bits contained in byte 5 ~ 12
            # voltage in byte 4 and 5, 0.1V resolution
            # current in byte 8 and 9, 0.1A resolution offset at 30000
            # soc in byte 10 and 11, 0.1% resolution
            # don't understand data in byte 6 and 7
            # below line shifts the MSB of first byte and combines with the second byte
            # as LSB to get the 16 bit int data
            voltage_data = ((data[4] << 8) | data[5]) / 10.0
            current_data = (((data[8] << 8) | data[9]) - 30000) / 10.0
            soc_data = ((data[10] << 8) | data[11]) / 10.0
            float_data = {
                'voltage': voltage_data,
                'current': current_data,
                'soc': soc_data
            }            
            return float_data
    elif code == 'temp':
        if data:
            # data is in 8 bits contained in byte 5 ~ 12
            # max temperature in byte 4 offset by 40C
            # max temperature cell no in byte 5
            # min temperature in byte 6
            # min temperature cell no in byte 7
            temp_data = {
                'max_temp': data[4] - 40,
                'max_temp_cell_no': data[5],
                'min_temp': data[6] - 40,
                'min_temp_cell_no': data[7]
            }
            return temp_data

safety_level = 0.8 # 80% safety level
# # temperature
degree_sign = u'\N{DEGREE SIGN}'
h_x100 = 22
h_x_tau = 90
sos_temp, m = calculate_sos(h_x_tau, h_x100, safety_level)

# # Discharge Current
h_x100 = 20
h_x_tau = 30
sos_current, m = calculate_sos(h_x_tau, h_x100, safety_level)

print("Starting to read data from BMS...")
while 1:
    temp_dict = get_cell_data('temp')
    sos_array = []
    if temp_dict:
        sos_val = sos_temp(temp_dict['max_temp'])
        sos_array.append(sos_val)
        print(f"Max Temp: {temp_dict['max_temp']}{degree_sign}C, "
              f"SOS for higher temp: {sos_val:.5f}")
    
    soc_dict = get_cell_data('soc')
    if soc_dict:
        sos_val = sos_current(soc_dict['current'])
        sos_array.append(sos_val)
        print(f"Current: {soc_dict['current']}A, "
              f"SOC: {soc_dict['soc']}%, "
              f"SOS for discharge current: {sos_val:.5f}")
        
    # multiply sos values to get overall sos
    if sos_array:
        overall_sos = 1
        for sos in sos_array:
            overall_sos *= sos
        print(f"Overall SOS: {overall_sos:.5f}")
    else:
        print("No data received or SOS values not calculated.")
    
    # break
    # battery_state = get_cell_data('battery_state')