const form = document.querySelector('form');
const nameInput = document.getElementById('name');
const emailInput = document.getElementById('email');
const phoneInput = document.getElementById('phone');
const passwordInput = document.getElementById('password');
const confirmPasswordInput = document.getElementById('confirm-password');
const submitButton = document.getElementById('submit');

form.addEventListener('submit', (event) => {
    event.preventDefault();
    let isValid = true;

    if (nameInput.value.length < 6) {
        nameInput.nextElementSibling.textContent = 'Name is too short';
        isValid = false;
    } else {
        nameInput.nextElementSibling.textContent = '';
    }

    if (!emailInput.value.includes('@')) {
        emailInput.nextElementSibling.textContent = 'Email must contain @ character';
        isValid = false;
    } else if (!emailInput.value.includes('.')) {
        emailInput.nextElementSibling.textContent = 'Email must contain . character';
        isValid = false;
    } else {
        emailInput.nextElementSibling.textContent = '';
    }

    if (phoneInput.value.length !== 10 || !/^\d+$/.test(phoneInput.value)) {
        phoneInput.nextElementSibling.textContent = 'Phone number must be 10 digits long and contain only digits';
        isValid = false;
    } else {
        phoneInput.nextElementSibling.textContent = '';
    }

    if (passwordInput.value !== confirmPasswordInput.value) {
        confirmPasswordInput.nextElementSibling.textContent = 'Passwords do not match';
        isValid = false;
    } else {
        confirmPasswordInput.nextElementSibling.textContent = '';
    }

    const passwordRegex = /^(?=.*[!@#$%^&*(),.?":{}|<>])(?=.*\d)(?=.*[A-Z])(?=.*[a-z]).{8,}$/;
    if (!passwordRegex.test(passwordInput.value)) {
        passwordInput.nextElementSibling.textContent = 'Password must contain a special character, a number, a capital letter, a small letter and be at least 8 characters long';
        isValid = false;
    } else {
        passwordInput.nextElementSibling.textContent = '';
    }

    if (isValid) {
        const confirmed = confirm('Are you sure you want to submit?');
        if (confirmed) {
          alert('Data submitted successfully');
          form.submit();
        } else {
          alert('Data not submitted');
        }
      }
    });

submitButton.onclick = () => {
    const confirmed = confirm('Are you sure you want to submit?');
    if (confirmed) {
      alert('Data submitted successfully');
    } else {
      alert('Data not submitted');
    }
  };